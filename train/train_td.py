import argparse
import os
import sys
import numpy as np
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.minigrid_env import make_env, MiniGridEncoder
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
    env = make_env(args.env_id, seed=args.seed)
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

        if args.method == "tdlambda":
            action = agent.select_action(state)
            while not terminated and not truncated:
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = encoder.encode(next_observation)
                next_action = (
                    agent.select_action(next_state)
                    if not (terminated or truncated)
                    else None
                )
                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done=terminated or truncated,
                    next_action=next_action,
                )
                state = next_state
                action = next_action if next_action is not None else action
                total_reward += reward
        else:
            while not terminated and not truncated:
                action = agent.select_action(state)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = encoder.encode(next_observation)
                agent.update(state, action, reward, next_state, done=terminated or truncated)
                state = next_state
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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    run_dir = os.path.join(args.save_dir, f"{timestamp}-{args.method}-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{args.method}_{safe_env_id}.npy")
    if hasattr(agent, "save"):
        agent.save(model_path)
    plot_learning_curve(
        history,
        title=f"{args.method.upper()} on {args.env_id}",
        save_path=os.path.join(run_dir, f"{args.method}_learning_curve.png"),
    )
    print(f"Saved trained model to {model_path}")
    print(f"Saved plot to {os.path.join(run_dir, f'{args.method}_learning_curve.png')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD(0) or TD(λ) on MiniGrid or Maze environments")
    parser.add_argument("--env-id", type=str, default="Maze-Easy")
    parser.add_argument("--method", type=str, default="tdlambda", choices=["td0", "tdlambda"])
    parser.add_argument("--num-episodes", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0, help="Starting epsilon for epsilon-greedy exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode")
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="saved_models")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
