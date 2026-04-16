import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.minigrid_env import make_env, MiniGridEncoder
from agents.q_learning import QLearningAgent
from utils.plot import plot_learning_curve


def run_training(args):

    env = make_env(args.env_id, seed=args.seed)
    encoder = MiniGridEncoder(env.observation_space)
    state_size = encoder.size
    n_actions = env.action_space.n

    agent = QLearningAgent(
        state_size=state_size,
        n_actions=n_actions,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    history = []

    for episode in range(1, args.num_episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        state = encoder.encode(obs)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encoder.encode(next_obs)
            agent.update(state, action, reward, next_state, done=terminated or truncated)
            state = next_state
            total_reward += reward

        history.append(total_reward)

        if args.epsilon_decay < 1.0:
            agent.epsilon = max(args.epsilon_min, agent.epsilon * args.epsilon_decay)

        if episode % args.log_interval == 0:
            mean_reward = np.mean(history[-args.log_interval:])
            print(
                f"Episode {episode}/{args.num_episodes}, "
                f"mean reward {mean_reward:.3f}, "
                f"last reward {total_reward:.3f}, "
                f"epsilon {agent.epsilon:.3f}"
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    run_dir = os.path.join(args.save_dir, f"{timestamp}-qlearning-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"qlearning_{safe_env_id}.npy")
    agent.save(model_path)

    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {
    "env_id": str(args.env_id),
    "algorithm": "qlearning",
    "num_episodes": int(args.num_episodes),
    "log_interval": int(args.log_interval),
    "state_size": int(state_size),
    "n_actions": int(n_actions),
    "alpha": float(args.alpha),
    "gamma": float(args.gamma),
    "epsilon_start": float(args.epsilon),
    "epsilon_min": float(args.epsilon_min),
    "epsilon_decay": float(args.epsilon_decay),
    "seed": int(args.seed),
    "save_dir": str(run_dir),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save episode rewards array
    reward_path = os.path.join(run_dir, "episode_rewards.npy")
    np.save(reward_path, np.array(history))
    print(f"Episode rewards saved to {reward_path}")

    plot_path = os.path.join(run_dir, "qlearning_learning_curve.png")
    plot_learning_curve(
        history,
        title=f"Q‑Learning on {args.env_id}",
        save_path=plot_path,
    )

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Learning curve saved to {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q‑Learning on MiniGrid or Maze environments")
    parser.add_argument("--env-id", type=str, default="Maze-Easy",
                        help="Environment ID (e.g., MiniGrid-Empty-8x8-v0, Maze-Easy)")
    parser.add_argument("--num-episodes", type=int, default=250,
                        help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial epsilon for epsilon‑greedy exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.05,
                        help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Multiplicative epsilon decay per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between console logs")
    parser.add_argument("--save-dir", type=str, default="saved_models",
                        help="Directory to save models and results")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
