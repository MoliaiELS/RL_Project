import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.ppo_agent import PPOAgent
from envs.minigrid_env import make_env
from utils.plot import plot_learning_curve

# Custom callback: record total reward per episode
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # After each step, accumulate the reward for the current episode
        self.current_episode_reward += self.locals["rewards"][0]
        # If the episode ends (done or truncated), save the reward and reset
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid or Maze")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0",
                        help="Environment ID (e.g., MiniGrid-Empty-8x8-v0, Maze-Easy)")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total number of timesteps to train")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate for PPO")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for PPO updates")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="saved_models",
                        help="Directory to save models and results")
    return parser.parse_args()


def run_training(args):
    # Create environment (to get observation space and action space info)
    env = make_env(args.env_id, seed=args.seed)
    # Create PPO agent
    agent = PPOAgent(
        env_id=args.env_id,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    agent.build(env=env)

    # Create callback to record rewards
    reward_callback = EpisodeRewardCallback()

    # Train
    agent.model.learn(
        total_timesteps=args.total_timesteps,
        callback=reward_callback,
    )

    # Get reward history
    history = reward_callback.episode_rewards
    if not history:
        print("Warning: No episode rewards recorded. Possibly no episode finished during training.")
        history = [0.0]

    # Prepare save directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    run_dir = os.path.join(args.save_dir, f"{timestamp}-ppo-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(run_dir, f"ppo_{safe_env_id}.zip")
    agent.save(model_path)

    # Save metadata
    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {
        "env_id": args.env_id,
        "algorithm": "ppo",
        "total_timesteps": int(args.total_timesteps),
        "gamma": float(args.gamma),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "save_dir": run_dir,
        "episodes_recorded": len(history),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Plot learning curve
    plot_path = os.path.join(run_dir, "ppo_learning_curve.png")
    plot_learning_curve(
        history,
        title=f"PPO on {args.env_id}",
        save_path=plot_path,
    )

    # Optional: also save reward array as npy
    reward_path = os.path.join(run_dir, "episode_rewards.npy")
    np.save(reward_path, np.array(history))

    print(f"PPO model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Learning curve saved to {plot_path}")
    print(f"Episode rewards saved to {reward_path}")


if __name__ == "__main__":
    run_training(parse_args())
