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
from agents.td_lambda_action_features import TDLambdaActionFeatureAgent
from utils.plot import plot_learning_curve


def parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one integer")
    try:
        return [int(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("List items must be integers") from exc


def build_agent(args, n_actions: int):
    return TDLambdaActionFeatureAgent(
        n_actions=n_actions,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        lambda_value=args.lambda_value,
        patch_radius=args.patch_radius,
        seed=args.seed,
    )


def evaluate_agent_greedy(agent, env, eval_episodes, seed):
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    for episode in range(1, eval_episodes + 1):
        observation, _ = env.reset(seed=seed + episode)
        if hasattr(agent, "new_episode"):
            agent.new_episode()
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


def normalize_schedule(stage_lengths: list[int], change_frequencies: list[int]) -> tuple[list[int], list[int]]:
    if len(stage_lengths) != len(change_frequencies):
        raise ValueError("stage_lengths and change_frequencies must have the same number of values")
    if any(value <= 0 for value in stage_lengths):
        raise ValueError("All stage lengths must be positive")
    if any(value <= 0 for value in change_frequencies):
        raise ValueError("All change frequencies must be positive")
    return stage_lengths, change_frequencies


def get_stage_index(episode: int, stage_boundaries: list[int]) -> int:
    for idx in range(len(stage_boundaries) - 1):
        if stage_boundaries[idx] < episode <= stage_boundaries[idx + 1]:
            return idx
    return len(stage_boundaries) - 2


def train_phase(
    agent,
    env_id: str,
    num_episodes: int,
    seed: int,
    use_manhattan_distance: bool,
    eval_interval: int,
    eval_episodes: int,
    log_interval: int,
    stage_lengths: list[int] | None,
    change_frequencies: list[int] | None,
    maze_change_frequency: int,
    epsilon_decay: float,
    epsilon_min: float,
    phase_name: str,
) -> tuple[list[float], list[tuple[int, float, float, float]]]:
    if stage_lengths is None or change_frequencies is None:
        stage_lengths = [num_episodes]
        change_frequencies = [maze_change_frequency]

    stage_lengths, change_frequencies = normalize_schedule(stage_lengths, change_frequencies)
    stage_boundaries = [0] + list(np.cumsum(stage_lengths))

    env = make_env(env_id, seed=seed, use_manhattan_distance=use_manhattan_distance)
    history: list[float] = []
    greedy_eval_history: list[tuple[int, float, float, float]] = []

    current_stage_index = -1
    current_block_index = -1
    current_layout_seed = None

    for episode in range(1, num_episodes + 1):
        stage_index = get_stage_index(episode, stage_boundaries)
        if stage_index != current_stage_index:
            current_stage_index = stage_index
            current_block_index = -1
            print(
                f"[{phase_name}] Stage {stage_index + 1}/{len(stage_lengths)} "
                f"length={stage_lengths[stage_index]} freq={change_frequencies[stage_index]}"
            )

        stage_start_episode = stage_boundaries[current_stage_index] + 1
        episode_in_stage = episode - stage_start_episode + 1
        frequency = change_frequencies[current_stage_index]
        block_index = (episode_in_stage - 1) // frequency
        if block_index != current_block_index:
            current_block_index = block_index
            current_layout_seed = seed + stage_boundaries[current_stage_index] + block_index

        observation, _ = env.reset(seed=current_layout_seed)
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

        if epsilon_decay < 1.0:
            agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        if episode % log_interval == 0:
            mean_reward = np.mean(history[-log_interval:])
            print(
                f"[{phase_name}] Episode {episode}/{num_episodes}, "
                f"mean reward {mean_reward:.3f}, last reward {total_reward:.3f}, "
                f"epsilon {agent.epsilon:.3f}"
            )

        if eval_interval > 0 and episode % eval_interval == 0:
            greedy_rewards = evaluate_agent_greedy(agent, env, eval_episodes, seed + episode)
            greedy_mean = float(np.mean(greedy_rewards))
            greedy_min = float(np.min(greedy_rewards))
            greedy_max = float(np.max(greedy_rewards))
            greedy_eval_history.append((episode, greedy_mean, greedy_min, greedy_max))
            print(
                f"[{phase_name}] Greedy eval @ {episode}: mean {greedy_mean:.3f}, "
                f"min {greedy_min:.3f}, max {greedy_max:.3f}"
            )

    return history, greedy_eval_history


def run_training(args):
    base_env_id = args.pretrain_env_id if args.pretrain_episodes > 0 else args.env_id
    base_env = make_env(base_env_id, seed=args.seed, use_manhattan_distance=args.use_manhattan_distance)
    agent = build_agent(args, n_actions=base_env.action_space.n)
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"Pretrained model not found: {args.load_model}")
        agent.load(args.load_model)
        print(f"Loaded pretrained model from {args.load_model}")

    history = []
    greedy_eval_history: list[tuple[int, float, float, float]] = []

    if args.pretrain_episodes > 0:
        pretrain_history, pretrain_eval = train_phase(
            agent=agent,
            env_id=args.pretrain_env_id,
            num_episodes=args.pretrain_episodes,
            seed=args.seed,
            use_manhattan_distance=args.use_manhattan_distance,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            stage_lengths=[args.pretrain_episodes],
            change_frequencies=[args.pretrain_episodes],
            maze_change_frequency=args.pretrain_episodes,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            phase_name="pretrain",
        )
        history.extend(pretrain_history)
        greedy_eval_history.extend(pretrain_eval)

    if args.num_episodes > 0:
        dynamic_history, dynamic_eval = train_phase(
            agent=agent,
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            seed=args.seed + args.pretrain_episodes,
            use_manhattan_distance=args.use_manhattan_distance,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            stage_lengths=args.stage_lengths,
            change_frequencies=args.change_frequencies,
            maze_change_frequency=args.maze_change_frequency,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            phase_name="random",
        )
        history.extend(dynamic_history)
        greedy_eval_history.extend(dynamic_eval)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    run_dir = os.path.join(ROOT_DIR, "saved_models", f"{timestamp}-tdlambda-actionfeatures-dynfreq-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"tdlambda_actionfeatures_{safe_env_id}.npy")
    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {
        "env_id": args.env_id,
        "pretrain_env_id": args.pretrain_env_id,
        "method": "tdlambda_actionfeatures",
        "pretrain_episodes": int(args.pretrain_episodes),
        "num_episodes": int(args.num_episodes),
        "log_interval": int(args.log_interval),
        "eval_interval": int(args.eval_interval),
        "eval_episodes": int(args.eval_episodes),
        "n_actions": int(agent.n_actions),
        "state_size": int(agent.state_size),
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "epsilon_min": float(args.epsilon_min),
        "epsilon_decay": float(args.epsilon_decay),
        "lambda_value": float(args.lambda_value),
        "use_manhattan_distance": bool(args.use_manhattan_distance),
        "patch_radius": int(args.patch_radius),
        "maze_change_frequency": int(args.maze_change_frequency),
        "stage_lengths": list(args.stage_lengths or [args.num_episodes]),
        "change_frequencies": list(args.change_frequencies or [args.maze_change_frequency]),
        "seed": int(args.seed),
        "save_dir": run_dir,
        "loaded_model": args.load_model,
    }

    agent.save(model_path)
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    plot_learning_curve(
        history,
        title=f"TD(λ) Action-Conditional Features on {args.env_id}",
        save_path=os.path.join(run_dir, f"tdlambda_actionfeatures_learning_curve.png"),
    )
    if greedy_eval_history:
        eval_episode_ids = [int(entry[0]) for entry in greedy_eval_history]
        eval_means = [entry[1] for entry in greedy_eval_history]
        plot_learning_curve(
            history,
            title=f"TD(λ) Action-Conditional Training and Greedy Eval on {args.env_id}",
            save_path=os.path.join(run_dir, f"tdlambda_actionfeatures_combined_learning_curve.png"),
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
        print(f"Saved greedy evaluation history to {greedy_summary_path}")

    print(f"Saved trained model to {model_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Saved plot to {os.path.join(run_dir, f'tdlambda_actionfeatures_learning_curve.png')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a TD(lambda) action-feature agent with dynamic maze-change frequency scheduling"
    )
    parser.add_argument("--pretrain-env-id", type=str, default="Maze-Auto")
    parser.add_argument("--pretrain-episodes", type=int, default=400)
    parser.add_argument("--env-id", type=str, default="Maze-Auto-Random-9x9")
    parser.add_argument("--num-episodes", type=int, default=1200)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Starting epsilon for epsilon-greedy exploration",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.08,
        help="Minimum epsilon after decay",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9998,
        help="Multiplicative epsilon decay per episode",
    )
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=2,
        help="Radius of the local patch used by the action-conditional feature extractor",
    )
    parser.add_argument(
        "--maze-change-frequency",
        type=int,
        default=10,
        help="How many episodes to keep the same random maze before changing it when no stage schedule is provided",
    )
    parser.add_argument(
        "--stage-lengths",
        type=parse_int_list,
        default="300,400,500",
        help="Comma-separated stage episode lengths for dynamic maze change scheduling",
    )
    parser.add_argument(
        "--change-frequencies",
        type=parse_int_list,
        default="50,20,1",
        help="Comma-separated maze change frequencies corresponding to stage lengths",
    )
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Run greedy evaluation every N episodes during training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of greedy evaluation episodes when eval_interval is enabled",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to a pretrained model file to load before training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default=os.path.join(ROOT_DIR, "saved_models"))
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
