import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.minigrid_env import make_env
from agents.td_lambda_graph import TDLambdaGraphAgent
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
    )


def select_env(env_ids: list[str], env_probs: list[float], rng: np.random.Generator) -> str:
    if len(env_ids) == 1:
        return env_ids[0]
    return str(rng.choice(env_ids, p=env_probs))


def select_seed(fixed_seeds: list[int] | None, seed_probs: list[float] | None, base_seed: int, episode: int, rng: np.random.Generator) -> int:
    if fixed_seeds is None:
        return base_seed + episode
    if seed_probs is None:
        seed_probs = [1.0 / len(fixed_seeds)] * len(fixed_seeds)
    return int(rng.choice(fixed_seeds, p=seed_probs))


def evaluate_agent_greedy(
    agent: TDLambdaGraphAgent,
    env_ids: list[str],
    env_probs: list[float],
    fixed_seeds: list[int] | None,
    seed_probs: list[float] | None,
    seed: int,
    eval_episodes: int,
    use_manhattan_distance: bool,
) -> list[float]:
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    rng = np.random.default_rng(seed)
    for episode in range(1, eval_episodes + 1):
        env_id = select_env(env_ids, env_probs, rng)
        env_seed = select_seed(fixed_seeds, seed_probs, seed, episode, rng)
        env = make_env(env_id, seed=env_seed, use_manhattan_distance=use_manhattan_distance)
        if hasattr(agent, "new_episode"):
            agent.new_episode()
        observation, _ = env.reset(seed=env_seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            action = agent.greedy_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    agent.epsilon = saved_epsilon
    return rewards


def run_training(args):
    env_ids = parse_list(args.env_ids) if args.env_ids else [args.env_id]
    env_probs = normalize_probs(parse_float_list(args.env_probs) if args.env_probs else [1.0] * len(env_ids), len(env_ids))
    fixed_seeds = parse_int_list(args.fixed_seeds) if args.fixed_seeds else None
    seed_probs = normalize_probs(parse_float_list(args.seed_probs) if args.seed_probs else [1.0] * len(fixed_seeds), len(fixed_seeds)) if fixed_seeds else None

    base_env = make_env(env_ids[0], seed=args.seed, use_manhattan_distance=args.use_manhattan_distance)
    agent = build_agent(args, n_actions=base_env.action_space.n)

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"Pretrained model not found: {args.load_model}")
        agent.load(args.load_model)
        print(f"Loaded pretrained model from {args.load_model}")

    history = []
    greedy_eval_history = []
    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.num_episodes + 1):
        env_id = select_env(env_ids, env_probs, rng)
        env_seed = select_seed(fixed_seeds, seed_probs, args.seed, episode, rng)
        env = make_env(env_id, seed=env_seed, use_manhattan_distance=args.use_manhattan_distance)

        observation, _ = env.reset(seed=env_seed)
        if hasattr(agent, "new_episode"):
            agent.new_episode()
        total_reward = 0.0
        terminated = False
        truncated = False

        action = agent.select_action(observation)
        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_action = (
                agent.select_action(next_observation)
                if not (terminated or truncated)
                else None
            )
            agent.update(
                observation,
                action,
                reward,
                next_observation,
                done=terminated or truncated,
                next_action=next_action,
            )
            observation = next_observation
            action = next_action if next_action is not None else action
            total_reward += reward

        history.append(total_reward)

        if args.epsilon_decay < 1.0:
            agent.epsilon = max(args.epsilon_min, agent.epsilon * args.epsilon_decay)

        if episode % args.log_interval == 0:
            mean_reward = np.mean(history[-args.log_interval :])
            print(
                f"Episode {episode}/{args.num_episodes}, mean reward {mean_reward:.3f}, "
                f"last reward {total_reward:.3f}, epsilon {agent.epsilon:.3f}"
            )

        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            greedy_rewards = evaluate_agent_greedy(
                agent,
                env_ids,
                env_probs,
                fixed_seeds,
                seed_probs,
                args.seed + episode,
                args.eval_episodes,
                args.use_manhattan_distance,
            )
            greedy_mean = float(np.mean(greedy_rewards))
            greedy_min = float(np.min(greedy_rewards))
            greedy_max = float(np.max(greedy_rewards))
            greedy_eval_history.append((episode, greedy_mean, greedy_min, greedy_max))
            print(
                f"Greedy eval @ {episode}: mean {greedy_mean:.3f}, "
                f"min {greedy_min:.3f}, max {greedy_max:.3f}"
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.save_name or "tdlambda_graph"
    safe_env_id = "_".join([env_id.replace('/', '_') for env_id in env_ids])
    run_dir = args.run_dir if args.run_dir else os.path.join(args.save_dir, f"{timestamp}-{name}-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{name}.pt")
    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {
        "env_ids": env_ids,
        "env_probs": env_probs,
        "fixed_seeds": fixed_seeds,
        "seed_probs": seed_probs,
        "method": "tdlambda_graph",
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
        "lambda_value": float(args.lambda_value),
        "use_manhattan_distance": bool(args.use_manhattan_distance),
        "seed": int(args.seed),
        "save_name": name,
        "save_dir": run_dir,
        "load_model": args.load_model,
    }

    agent.save(model_path)
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    reward_path = os.path.join(run_dir, "episode_rewards.npy")
    np.save(reward_path, np.array(history))

    plot_learning_curve(
        history,
        title=f"TD(λ) Graph Agent on {safe_env_id}",
        save_path=os.path.join(run_dir, f"{name}_learning_curve.png"),
    )
    if greedy_eval_history:
        eval_episode_ids = [int(entry[0]) for entry in greedy_eval_history]
        eval_means = [entry[1] for entry in greedy_eval_history]
        plot_learning_curve(
            history,
            title=f"TD(λ) Graph Training and Greedy Eval on {safe_env_id}",
            save_path=os.path.join(run_dir, f"{name}_combined_learning_curve.png"),
            secondary_rewards=eval_means,
            secondary_x=eval_episode_ids,
            secondary_label="Greedy eval mean",
        )
        greedy_summary_path = os.path.join(run_dir, "greedy_eval_history.json")
        with open(greedy_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eval_interval": args.eval_interval,
                    "eval_episodes": args.eval_episodes,
                    "history": [
                        {"episode": int(e), "mean": float(m), "min": float(mi), "max": float(ma)}
                        for e, m, mi, ma in greedy_eval_history
                    ],
                },
                f,
                indent=2,
            )

    print(f"Episode rewards saved to {reward_path}")
    print(f"Saved trained model to {model_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Saved plot to {os.path.join(run_dir, f'{name}_learning_curve.png')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a graph-based TD(lambda) maze agent")
    parser.add_argument("--env-id", type=str, default="Maze-Auto-Random-9x9")
    parser.add_argument(
        "--env-ids",
        type=parse_list,
        default=None,
        help="Comma-separated environment IDs to sample from during training.",
    )
    parser.add_argument(
        "--env-probs",
        type=parse_float_list,
        default=None,
        help="Comma-separated probabilities for --env-ids.",
    )
    parser.add_argument(
        "--fixed-seeds",
        type=parse_int_list,
        default=None,
        help="Comma-separated maze seeds to sample from for deterministic random-graph pretraining.",
    )
    parser.add_argument(
        "--seed-probs",
        type=parse_float_list,
        default=None,
        help="Comma-separated seed probabilities for --fixed-seeds.",
    )
    parser.add_argument("--num-episodes", type=int, default=800)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-name", type=str, default="tdlambda_graph")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default=os.path.join(ROOT_DIR, "saved_models"))
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use, such as cpu or cuda.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
