from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.td_lambda_graph import TDLambdaGraphAgent
from envs.minigrid_env import make_env
from utils.plot import plot_learning_curve


def parse_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        if not value:
            raise argparse.ArgumentTypeError("List must contain at least one value")
        return value
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one value")
    return values


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        if not value:
            raise argparse.ArgumentTypeError("List must contain at least one value")
        return value
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one value")
    try:
        return [float(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List items must be floats") from exc


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        if not value:
            raise argparse.ArgumentTypeError("List must contain at least one integer")
        return value
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one integer")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List items must be integers") from exc


def normalize_probs(probs: list[float], count: int) -> list[float]:
    if len(probs) != count:
        raise ValueError(f"Expected {count} probability values, got {len(probs)}")
    if any(p < 0.0 for p in probs):
        raise ValueError("Probabilities must be non-negative")
    total = float(sum(probs))
    if total <= 0.0:
        raise ValueError("Probability values must sum to a positive number")
    return [p / total for p in probs]


def running_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


def build_agent(args, n_actions: int) -> TDLambdaGraphAgent:
    return TDLambdaGraphAgent(
        n_actions=n_actions,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        lambda_value=args.lambda_value,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        use_target_network=args.use_target_network,
        target_tau=args.target_tau,
        target_update_freq=args.target_update_freq,
        batch_size=args.batch_size,
        gradient_clip=args.gradient_clip,
    )


def select_env(env_ids: list[str], env_probs: list[float], rng: np.random.Generator) -> str:
    if len(env_ids) == 1:
        return env_ids[0]
    return str(rng.choice(env_ids, p=env_probs))


def select_seed(
    fixed_seeds: list[int] | None,
    seed_probs: list[float] | None,
    base_seed: int,
    episode: int,
    rng: np.random.Generator,
) -> int:
    if fixed_seeds is None:
        return base_seed + episode
    if seed_probs is None:
        seed_probs = [1.0 / len(fixed_seeds)] * len(fixed_seeds)
    return int(rng.choice(fixed_seeds, p=seed_probs))


def rollout_episode(
    agent: TDLambdaGraphAgent,
    env,
    env_seed: int,
) -> tuple[list[dict], dict]:
    try:
        observation, _ = env.reset(seed=env_seed)
        agent.new_episode()

        total_reward = 0.0
        steps = 0
        invalid_action_count = 0
        fallback_action_count = 0
        transitions: list[dict] = []

        action, action_info = agent.select_action_details(observation, greedy=False)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            invalid_action_count += int(not action_info["selected_valid"])
            fallback_action_count += int(action_info["used_fallback"])

            if done:
                next_action = None
                next_action_info = None
            else:
                next_action, next_action_info = agent.select_action_details(next_observation, greedy=False)

            transitions.append(
                agent.update(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done=done,
                    next_action=next_action,
                )
            )

            observation = next_observation
            if next_action is None:
                break
            action = next_action
            action_info = next_action_info

        success = bool(getattr(env, "agent_pos", None) == getattr(env, "goal_pos", object()))
        return transitions, {
            "reward": float(total_reward),
            "episode_length": int(steps),
            "success": float(success),
            "invalid_action_rate": float(invalid_action_count / max(1, steps)),
            "fallback_action_rate": float(fallback_action_count / max(1, steps)),
        }
    finally:
        env.close()


def evaluate_agent_greedy(
    agent: TDLambdaGraphAgent,
    env_ids: list[str],
    env_probs: list[float],
    fixed_seeds: list[int] | None,
    seed_probs: list[float] | None,
    seed: int,
    eval_episodes: int,
    use_manhattan_distance: bool,
) -> dict:
    rewards = []
    lengths = []
    successes = 0
    rng = np.random.default_rng(seed)

    for episode in range(1, eval_episodes + 1):
        env_id = select_env(env_ids, env_probs, rng)
        env_seed = select_seed(fixed_seeds, seed_probs, seed, episode, rng)
        env = make_env(env_id, seed=env_seed, use_manhattan_distance=use_manhattan_distance)

        try:
            observation, _ = env.reset(seed=env_seed)
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = agent.select_action_details(observation, greedy=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            success = bool(getattr(env, "agent_pos", None) == getattr(env, "goal_pos", object()))
            rewards.append(float(total_reward))
            lengths.append(int(steps))
            successes += int(success)
        finally:
            env.close()

    failure_count = eval_episodes - successes
    return {
        "mean": float(np.mean(rewards)) if rewards else 0.0,
        "min": float(np.min(rewards)) if rewards else 0.0,
        "max": float(np.max(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_count": int(successes),
        "failure_count": int(failure_count),
        "rewards": rewards,
    }


def maybe_decay_epsilon(agent: TDLambdaGraphAgent, args, episode: int) -> None:
    if episode <= args.epsilon_hold_episodes:
        return
    if args.epsilon_decay < 1.0:
        agent.epsilon = max(args.epsilon_min, agent.epsilon * args.epsilon_decay)


def run_training(args) -> tuple[TDLambdaGraphAgent, list[dict], list[dict]]:
    env_ids = parse_list(args.env_ids) if args.env_ids else [args.env_id]
    env_probs = normalize_probs(
        parse_float_list(args.env_probs) if args.env_probs else [1.0] * len(env_ids),
        len(env_ids),
    )
    fixed_seeds = parse_int_list(args.fixed_seeds) if args.fixed_seeds else None
    seed_probs = (
        normalize_probs(
            parse_float_list(args.seed_probs) if args.seed_probs else [1.0] * len(fixed_seeds),
            len(fixed_seeds),
        )
        if fixed_seeds
        else None
    )

    base_env = make_env(env_ids[0], seed=args.seed, use_manhattan_distance=args.use_manhattan_distance)
    try:
        agent = build_agent(args, n_actions=base_env.action_space.n)
    finally:
        base_env.close()

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"Pretrained model not found: {args.load_model}")
        agent.load(args.load_model)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = args.alpha
        print(f"Loaded pretrained model from {args.load_model}")

    metrics_history: list[dict] = []
    greedy_eval_history: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.num_episodes + 1):
        env_id = select_env(env_ids, env_probs, rng)
        env_seed = select_seed(fixed_seeds, seed_probs, args.seed, episode, rng)
        env = make_env(env_id, seed=env_seed, use_manhattan_distance=args.use_manhattan_distance)
        transitions, episode_stats = rollout_episode(agent, env, env_seed)
        update_metrics = agent.update_from_episode(transitions)

        reward_history = [entry["episode_reward"] for entry in metrics_history] + [episode_stats["reward"]]
        metrics_entry = {
            "episode": int(episode),
            "env_id": env_id,
            "env_seed": int(env_seed),
            "episode_reward": episode_stats["reward"],
            "running_mean_reward": running_mean(reward_history, args.log_interval),
            "success_rate": episode_stats["success"],
            "episode_length": episode_stats["episode_length"],
            "invalid_action_rate": episode_stats["invalid_action_rate"],
            "fallback_action_rate": episode_stats["fallback_action_rate"],
            "mean_td_target": update_metrics["mean_td_target"],
            "mean_predicted_q": update_metrics["mean_predicted_q"],
            "loss": update_metrics["loss"],
            "epsilon": float(agent.epsilon),
            "optimizer_steps": int(update_metrics["optimizer_steps"]),
        }
        metrics_history.append(metrics_entry)

        maybe_decay_epsilon(agent, args, episode)
        metrics_entry["epsilon_after_schedule"] = float(agent.epsilon)

        if episode % args.log_interval == 0:
            window = metrics_history[-args.log_interval :]
            print(
                f"Episode {episode:5d} | Avg Reward: {np.mean([m['episode_reward'] for m in window]):7.3f} | "
                f"Run Mean: {metrics_entry['running_mean_reward']:7.3f} | "
                f"Success: {np.mean([m['success_rate'] for m in window]):.2f} | "
                f"Avg Len: {np.mean([m['episode_length'] for m in window]):6.2f} | "
                f"Invalid: {np.mean([m['invalid_action_rate'] for m in window]):.3f} | "
                f"Avg Target: {np.mean([m['mean_td_target'] for m in window]):7.3f} | "
                f"Avg Q: {np.mean([m['mean_predicted_q'] for m in window]):7.3f} | "
                f"Avg Loss: {np.mean([m['loss'] for m in window]):8.5f} | "
                f"Epsilon: {agent.epsilon:.4f} | Env: {env_id}"
            )

        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            eval_summary = evaluate_agent_greedy(
                agent,
                env_ids,
                env_probs,
                fixed_seeds,
                seed_probs,
                args.seed + episode,
                args.eval_episodes,
                args.use_manhattan_distance,
            )
            eval_summary["episode"] = int(episode)
            greedy_eval_history.append(eval_summary)
            print(
                f"Greedy eval @ {episode}: mean {eval_summary['mean']:.3f} | "
                f"min {eval_summary['min']:.3f} | max {eval_summary['max']:.3f} | "
                f"success {eval_summary['success_count']}/{args.eval_episodes}"
            )

    return agent, metrics_history, greedy_eval_history


def save_run_outputs(
    args,
    agent: TDLambdaGraphAgent,
    metrics_history: list[dict],
    greedy_eval_history: list[dict],
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_ids = parse_list(args.env_ids) if args.env_ids else [args.env_id]
    env_probs = normalize_probs(
        parse_float_list(args.env_probs) if args.env_probs else [1.0] * len(env_ids),
        len(env_ids),
    )
    fixed_seeds = parse_int_list(args.fixed_seeds) if args.fixed_seeds else None
    seed_probs = (
        normalize_probs(
            parse_float_list(args.seed_probs) if args.seed_probs else [1.0] * len(fixed_seeds),
            len(fixed_seeds),
        )
        if fixed_seeds
        else None
    )

    name = args.save_name or "tdlambda_graph"
    safe_env_id = "_".join(env_id.replace("/", "_") for env_id in env_ids)
    run_dir = args.run_dir or os.path.join(args.save_dir, f"{timestamp}-{name}-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{name}.pt")
    metadata_path = os.path.join(run_dir, "metadata.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    eval_path = os.path.join(run_dir, "greedy_eval_history.json")
    summary_path = os.path.join(run_dir, "stage_summary.json")

    agent.save(model_path)

    rewards = [entry["episode_reward"] for entry in metrics_history]
    losses = [entry["loss"] for entry in metrics_history]
    eval_means = [entry["mean"] for entry in greedy_eval_history]
    eval_episodes = [entry["episode"] for entry in greedy_eval_history]

    metadata = {
        "method": "tdlambda_graph_lambda_return",
        "update_mode": agent.update_mode,
        "env_ids": env_ids,
        "env_probs": env_probs,
        "fixed_seeds": fixed_seeds,
        "seed_probs": seed_probs,
        "num_episodes": int(args.num_episodes),
        "log_interval": int(args.log_interval),
        "eval_interval": int(args.eval_interval),
        "eval_episodes": int(args.eval_episodes),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "n_actions": int(agent.n_actions),
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "epsilon_min": float(args.epsilon_min),
        "epsilon_decay": float(args.epsilon_decay),
        "epsilon_hold_episodes": int(args.epsilon_hold_episodes),
        "lambda_value": float(args.lambda_value),
        "batch_size": int(args.batch_size),
        "gradient_clip": float(args.gradient_clip),
        "use_target_network": bool(args.use_target_network),
        "target_tau": float(args.target_tau),
        "target_update_freq": int(args.target_update_freq),
        "use_manhattan_distance": bool(args.use_manhattan_distance),
        "seed": int(args.seed),
        "device": str(agent.device),
        "save_name": name,
        "save_dir": run_dir,
        "load_model": args.load_model,
        "stage_name": args.stage_name,
    }

    summary = {
        "stage_name": args.stage_name or name,
        "num_episodes": int(args.num_episodes),
        "final_running_mean_reward": metrics_history[-1]["running_mean_reward"] if metrics_history else 0.0,
        "final_success_rate": metrics_history[-1]["success_rate"] if metrics_history else 0.0,
        "final_episode_length": metrics_history[-1]["episode_length"] if metrics_history else 0.0,
        "final_invalid_action_rate": metrics_history[-1]["invalid_action_rate"] if metrics_history else 0.0,
        "final_loss": metrics_history[-1]["loss"] if metrics_history else 0.0,
        "best_eval_mean": max(eval_means) if eval_means else None,
        "best_eval_episode": eval_episodes[int(np.argmax(eval_means))] if eval_means else None,
        "final_epsilon": metrics_history[-1]["epsilon_after_schedule"] if metrics_history else args.epsilon,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "eval_path": eval_path,
    }

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_history, metrics_file, indent=2)
    with open(eval_path, "w", encoding="utf-8") as eval_file:
        json.dump(greedy_eval_history, eval_file, indent=2)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    plot_learning_curve(
        rewards,
        window=max(10, min(args.log_interval, 100)),
        title="Training Reward Curve",
        save_path=os.path.join(run_dir, "reward_curve.png"),
        primary_label="Episode reward",
        ylabel="Reward",
    )
    plot_learning_curve(
        losses,
        window=max(10, min(args.log_interval, 100)),
        title="Training Loss Curve",
        save_path=os.path.join(run_dir, "training_loss_curve.png"),
        primary_label="Episode loss",
        ylabel="Loss",
    )
    if greedy_eval_history:
        plot_learning_curve(
            eval_means,
            window=max(1, min(len(eval_means), 5)),
            title="Greedy Evaluation Curve",
            save_path=os.path.join(run_dir, "eval_curve.png"),
            x_values=eval_episodes,
            primary_label="Greedy eval mean reward",
            ylabel="Reward",
        )

    plot_learning_curve(
        rewards,
        window=max(10, min(args.log_interval, 100)),
        title="Training and Greedy Evaluation",
        save_path=os.path.join(run_dir, "combined_curve.png"),
        primary_label="Episode reward",
        secondary_rewards=eval_means if greedy_eval_history else None,
        secondary_x=eval_episodes if greedy_eval_history else None,
        secondary_label="Greedy eval mean",
        ylabel="Reward",
    )

    return {
        "run_dir": run_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "eval_path": eval_path,
        "summary_path": summary_path,
    }


def build_parser(description: str = "Train a graph-based TD(lambda) maze agent") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--env-id", type=str, default="Maze-Auto-Random-9x9")
    parser.add_argument("--env-ids", type=parse_list, default=None, help="Comma-separated environment IDs.")
    parser.add_argument("--env-probs", type=parse_float_list, default=None, help="Comma-separated probabilities for --env-ids.")
    parser.add_argument("--fixed-seeds", type=parse_int_list, default=None, help="Comma-separated fixed maze seeds.")
    parser.add_argument("--seed-probs", type=parse_float_list, default=None, help="Comma-separated probabilities for --fixed-seeds.")
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-min", type=float, default=0.08)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--epsilon-hold-episodes", type=int, default=0)
    parser.add_argument("--lambda-value", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--use-target-network",
        action="store_true",
        dest="use_target_network",
        default=True,
        help="Use a target network for bootstrap stabilization.",
    )
    parser.add_argument(
        "--no-target-network",
        action="store_false",
        dest="use_target_network",
        help="Disable the target network.",
    )
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-freq", type=int, default=1)
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-name", type=str, default="tdlambda_graph")
    parser.add_argument("--stage-name", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default=os.path.join(ROOT_DIR, "saved_models"))
    parser.add_argument("--device", type=str, default=None, help="Torch device such as cpu or cuda.")
    return parser


def parse_args(description: str = "Train a graph-based TD(lambda) maze agent"):
    return build_parser(description).parse_args()


def cleanup_runtime() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    try:
        import pygame

        pygame.quit()
    except Exception:
        pass


def main(description: str = "Train a graph-based TD(lambda) maze agent") -> None:
    args = parse_args(description)
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        print("Training graph-based TD(lambda) agent")
        print(f"Environment: {args.env_id if args.env_ids is None else ','.join(parse_list(args.env_ids))}")
        print(f"Device: {args.device or 'auto'}")
        print(f"Save name: {args.save_name}")
        if args.stage_name:
            print(f"Stage: {args.stage_name}")
        print()

        agent, metrics_history, greedy_eval_history = run_training(args)
        outputs = save_run_outputs(args, agent, metrics_history, greedy_eval_history)

        print(f"Saved trained model to {outputs['model_path']}")
        print(f"Saved metadata to {outputs['metadata_path']}")
        print(f"Saved training metrics to {outputs['metrics_path']}")
        print(f"Saved evaluation history to {outputs['eval_path']}")
        print(f"Saved stage summary to {outputs['summary_path']}")
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
