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
from agents.td_lambda_graph_gpu import TDLambdaGraphAgentGPU
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


def build_agent(args, n_actions: int) -> TDLambdaGraphAgentGPU:
    return TDLambdaGraphAgentGPU(
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
    agent: TDLambdaGraphAgentGPU,
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

        if hasattr(agent, "new_episode"):
            agent.new_episode()

        observation, _ = env.reset(seed=env_seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not terminated and not truncated:
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                next_action = None
            else:
                next_action = agent.argmax_action(agent.q_values(next_observation))

            agent.update(observation, action, reward, next_observation, terminated or truncated, next_action)
            observation = next_observation

        history.append({
            "episode": episode,
            "env_id": env_id,
            "env_seed": env_seed,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": agent.epsilon,
        })

        if episode % args.log_interval == 0:
            recent_rewards = [h["total_reward"] for h in history[-args.log_interval:]]
            print(f"Episode {episode:4d} | Avg Reward: {np.mean(recent_rewards):6.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | Env: {env_id}")

        if episode % args.eval_interval == 0:
            eval_rewards = evaluate_agent_greedy(
                agent, env_ids, env_probs, fixed_seeds, seed_probs,
                args.seed, args.eval_episodes, args.use_manhattan_distance
            )
            avg_eval_reward = np.mean(eval_rewards)
            greedy_eval_history.append({
                "episode": episode,
                "avg_reward": avg_eval_reward,
                "eval_rewards": eval_rewards,
            })
            print(f"Evaluation at episode {episode}: Avg Reward = {avg_eval_reward:.2f}")

    return history, greedy_eval_history


def main():
    parser = argparse.ArgumentParser(description="Train GPU-accelerated graph-based TD(lambda) agent")
    parser.add_argument("--env-id", type=str, default="Maze-Easy", help="Environment ID")
    parser.add_argument("--env-ids", type=str, default=None, help="Comma-separated list of environment IDs")
    parser.add_argument("--env-probs", type=str, default=None, help="Comma-separated list of environment probabilities")
    parser.add_argument("--fixed-seeds", type=str, default=None, help="Comma-separated list of fixed environment seeds")
    parser.add_argument("--seed-probs", type=str, default=None, help="Comma-separated list of seed probabilities")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy")
    parser.add_argument("--lambda-value", type=float, default=0.9, help="TD(lambda) lambda parameter")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for graph encoder")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of graph encoder layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging interval")
    parser.add_argument("--save-name", type=str, default=None, help="Model save name")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )

    args = parser.parse_args()

    if args.save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_name = f"{timestamp}-tdlambda-gpu-graph-{args.env_id.replace('Maze-', '').lower()}"

    if args.run_dir is None:
        args.run_dir = os.path.join(ROOT_DIR, "saved_models", args.save_name)

    os.makedirs(args.run_dir, exist_ok=True)

    print(f"Training GPU-accelerated graph-based TD(lambda) agent")
    print(f"Environment: {args.env_id}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Save directory: {args.run_dir}")
    print(f"Model name: {args.save_name}")
    print()

    history, greedy_eval_history = run_training(args)

    # Save model
    model_path = os.path.join(args.run_dir, f"{args.save_name}.pt")
    agent.save(model_path)
    print(f"Saved model to {model_path}")

    # Save training history
    history_path = os.path.join(args.run_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save evaluation history
    eval_history_path = os.path.join(args.run_dir, "greedy_eval_history.json")
    with open(eval_history_path, "w", encoding="utf-8") as f:
        json.dump(greedy_eval_history, f, indent=2)
    print(f"Saved evaluation history to {eval_history_path}")

    # Save metadata
    metadata = {
        "agent_type": "TDLambdaGraphAgentGPU",
        "env_id": args.env_id,
        "env_ids": parse_list(args.env_ids) if args.env_ids else [args.env_id],
        "env_probs": parse_float_list(args.env_probs) if args.env_probs else None,
        "fixed_seeds": parse_int_list(args.fixed_seeds) if args.fixed_seeds else None,
        "seed_probs": parse_float_list(args.seed_probs) if args.seed_probs else None,
        "num_episodes": args.num_episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "lambda_value": args.lambda_value,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "device": args.device,
        "use_manhattan_distance": args.use_manhattan_distance,
        "model_path": model_path,
        "history_path": history_path,
        "eval_history_path": eval_history_path,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(args.run_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Plot learning curve
    plot_learning_curve(history, greedy_eval_history, save_path=os.path.join(args.run_dir, "learning_curve.png"))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()