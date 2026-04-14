import argparse
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.ppo_agent import PPOAgent
from envs.minigrid_env import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="saved_models/ppo_model")
    return parser.parse_args()


def run_training(args):
    agent = PPOAgent(
        env_id=args.env_id,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    env = make_env(args.env_id, seed=args.seed)
    agent.build(env=env)
    agent.train(total_timesteps=args.total_timesteps)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    base_dir = os.path.dirname(args.save_path) or "saved_models"
    run_dir = os.path.join(base_dir, f"{timestamp}-ppo-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)
    model_filename = os.path.basename(args.save_path) or "ppo_model"
    model_path = os.path.join(run_dir, model_filename)
    agent.save(model_path)
    print(f"PPO model saved to {model_path}")


if __name__ == "__main__":
    run_training(parse_args())
