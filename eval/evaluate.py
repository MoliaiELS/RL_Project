import argparse
import numpy as np
from envs.minigrid_env import make_minigrid_env, MiniGridEncoder
from agents.td_zero import TDZeroAgent
from agents.td_lambda import TDLambdaAgent


def load_td_agent(agent_type: str, state_size: int, n_actions: int, path: str, args):
    if agent_type == "td0":
        agent = TDZeroAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=0.0,
            seed=args.seed,
        )
    elif agent_type == "tdlambda":
        agent = TDLambdaAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=0.0,
            lambda_value=args.lambda_value,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    agent.load(path)
    return agent


def evaluate(args):
    env = make_minigrid_env(args.env_id, seed=args.seed)
    encoder = MiniGridEncoder(env.observation_space)
    agent = load_td_agent(args.agent_type, encoder.size, env.action_space.n, args.model_path, args)

    results = []
    for episode in range(1, args.eval_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        state = encoder.encode(observation)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action = agent.greedy_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            state = encoder.encode(observation)
            total_reward += reward

        results.append(total_reward)
        print(f"Eval episode {episode}: reward={total_reward:.3f}")

    mean_reward = float(np.mean(results))
    print(f"Average evaluation reward over {args.eval_episodes} episodes: {mean_reward:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TD agent")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--agent-type", type=str, default="tdlambda", choices=["td0", "tdlambda"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--lambda-value", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
