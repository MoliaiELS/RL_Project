import argparse
import os
import numpy as np
from envs.minigrid_env import make_minigrid_env, MiniGridEncoder
from agents.td_lambda import TDLambdaAgent
from agents.td_zero import TDZeroAgent
from utils.plot import plot_learning_curve


def build_agent(method: str, state_size: int, n_actions: int, args):
    if method == "td0":
        return TDZeroAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=args.epsilon,
            seed=args.seed,
        )
    if method == "tdlambda":
        return TDLambdaAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=args.epsilon,
            lambda_value=args.lambda_value,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported method: {method}")


def run_training(args):
    env = make_minigrid_env(args.env_id, seed=args.seed)
    encoder = MiniGridEncoder(env.observation_space)
    agent = build_agent(args.method, encoder.size, env.action_space.n, args)

    history = []
    for episode in range(1, args.num_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        state = encoder.encode(observation)
        if hasattr(agent, "new_episode"):
            agent.new_episode()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = encoder.encode(next_observation)
            agent.update(state, action, reward, next_state, done=terminated or truncated)
            state = next_state
            total_reward += reward

        history.append(total_reward)
        if episode % args.log_interval == 0:
            mean_reward = np.mean(history[-args.log_interval :])
            print(
                f"Episode {episode}/{args.num_episodes}, mean reward {mean_reward:.3f}, "
                f"last reward {total_reward:.3f}"
            )

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{args.method}.npy")
    if hasattr(agent, "save"):
        agent.save(model_path)
    plot_learning_curve(history, title=f"{args.method.upper()} on {args.env_id}", save_path=os.path.join(args.save_dir, f"{args.method}_learning_curve.png"))
    print(f"Saved trained model to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD(0) or TD(λ) on MiniGrid")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--method", type=str, default="tdlambda", choices=["td0", "tdlambda"])
    parser.add_argument("--num-episodes", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--lambda-value", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="saved_models")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
