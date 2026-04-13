import argparse
from agents.ppo_agent import PPOAgent


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
    agent.build()
    agent.train(total_timesteps=args.total_timesteps)
    agent.save(args.save_path)
    print(f"PPO model saved to {args.save_path}")


if __name__ == "__main__":
    run_training(parse_args())
