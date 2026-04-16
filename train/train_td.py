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


def evaluate_agent_greedy(agent, env, encoder, eval_episodes, seed):
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    for episode in range(1, eval_episodes + 1):
        observation, _ = env.reset(seed=seed + episode)
        state = encoder.encode(observation)
        if hasattr(agent, "new_episode"):
            agent.new_episode()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action = agent.greedy_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            state = encoder.encode(observation)
            total_reward += reward

        rewards.append(total_reward)

    agent.epsilon = saved_epsilon
    return rewards


def run_training(args):
    env = make_env(
        args.env_id,
        seed=args.seed,
        use_manhattan_distance=args.use_manhattan_distance,
    )
    encoder = MiniGridEncoder(env.observation_space)
    agent = build_agent(args.method, encoder.size, env.action_space.n, args)
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"Pretrained model not found: {args.load_model}")
        agent.load(args.load_model)
        print(f"Loaded pretrained model from {args.load_model}")

    history = []
    greedy_eval_history = []
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

        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            greedy_rewards = evaluate_agent_greedy(
                agent,
                env,
                encoder,
                args.eval_episodes,
                args.seed + episode,
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
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    run_dir = os.path.join(args.save_dir, f"{timestamp}-{args.method}-{safe_env_id}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, f"{args.method}_{safe_env_id}.npy")
    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {
        "env_id": args.env_id,
        "method": args.method,
        "num_episodes": int(args.num_episodes),
        "log_interval": int(args.log_interval),
        "eval_interval": int(args.eval_interval),
        "eval_episodes": int(args.eval_episodes),
        "state_size": int(encoder.size),
        "n_actions": int(env.action_space.n),
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "epsilon_min": float(args.epsilon_min),
        "epsilon_decay": float(args.epsilon_decay),
        "lambda_value": float(args.lambda_value),
        "use_manhattan_distance": bool(args.use_manhattan_distance),
        "seed": int(args.seed),
        "save_dir": run_dir,
        "loaded_model": args.load_model,
    }
    if hasattr(agent, "save"):
        agent.save(model_path)
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    # Save episode rewards array
    reward_path = os.path.join(run_dir, "episode_rewards.npy")
    np.save(reward_path, np.array(history))
    print(f"Episode rewards saved to {reward_path}")

    plot_learning_curve(
        history,
        title=f"{args.method.upper()} on {args.env_id}",
        save_path=os.path.join(run_dir, f"{args.method}_learning_curve.png"),
    )
    if greedy_eval_history:
        eval_episode_ids = [int(entry[0]) for entry in greedy_eval_history]
        eval_means = [entry[1] for entry in greedy_eval_history]
        eval_mins = [entry[2] for entry in greedy_eval_history]
        eval_maxs = [entry[3] for entry in greedy_eval_history]
        plot_learning_curve(
            history,
            title=f"TD(λ) training and greedy eval on {args.env_id}",
            save_path=os.path.join(run_dir, f"{args.method}_combined_learning_curve.png"),
            secondary_rewards=eval_means,
            secondary_x=eval_episode_ids,
            secondary_label="Greedy eval mean",
        )
        plot_learning_curve(
            eval_means,
            title=f"Greedy eval mean on {args.env_id}",
            save_path=os.path.join(run_dir, f"{args.method}_greedy_eval_curve.png"),
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
        print(f"Saved greedy evaluation curve to {os.path.join(run_dir, f'{args.method}_greedy_eval_curve.png')}")
        print(f"Saved greedy evaluation history to {greedy_summary_path}")
    print(f"Saved trained model to {model_path}")
    print(f"Saved metadata to {meta_path}")
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
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument("--eval-interval", type=int, default=0, help="Run greedy evaluation every N episodes during training")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of greedy evaluation episodes when eval_interval is enabled")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a pretrained model file to load before training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="saved_models")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
