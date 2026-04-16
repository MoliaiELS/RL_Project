import argparse
import glob
import json
import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.minigrid_env import make_env, MiniGridEncoder
from agents.td_zero import TDZeroAgent
from agents.td_lambda import TDLambdaAgent
from agents.td_lambda_action_features import TDLambdaActionFeatureAgent
from agents.td_lambda_cnn import TDLambdaCNNAgent


def _serialize_args(args):
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def save_evaluation_summary(rewards, model_path: str, args, model_metadata=None):
    model_dir = os.path.dirname(model_path) or "."
    summary = {
        "env_id": args.env_id,
        "agent_type": args.agent_type,
        "model_path": args.model_path,
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "args": _serialize_args(args),
        "model_metadata": model_metadata or {},
        "rewards": [float(r) for r in rewards],
        "mean_reward": float(np.mean(rewards)) if rewards else None,
    }
    summary_path = os.path.join(model_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_path}")
    return summary_path


def save_evaluation_plot(rewards, model_path: str, env_id: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to save evaluation plots")
        return None

    model_dir = os.path.dirname(model_path) or "."
    plot_path = os.path.join(model_dir, "eval_rewards.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, label="Episode reward", alpha=0.6)
    if len(rewards) >= 2:
        window = min(10, len(rewards))
        smoothed = [
            sum(rewards[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
            for i in range(len(rewards))
        ]
        ax.plot(smoothed, label=f"Running mean ({window})", color="tab:red")
    ax.set_title(f"Evaluation Rewards on {env_id}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation plot to {plot_path}")
    return plot_path


def load_td_agent(agent_type: str, state_size: int | None, n_actions: int, path: str, args):
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
    elif agent_type == "tdlambda_actionfeatures":
        agent = TDLambdaActionFeatureAgent(
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=0.0,
            lambda_value=args.lambda_value,
            patch_radius=args.patch_radius,
            seed=args.seed,
        )
    elif agent_type == "tdlambda_cnn":
        obs_shape = getattr(args, "obs_shape", None)
        if obs_shape is None:
            raise ValueError("obs_shape metadata is required to load tdlambda_cnn models.")
        agent = TDLambdaCNNAgent(
            obs_shape=tuple(obs_shape),
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


def load_model_metadata(model_path: str) -> dict:
    model_dir = os.path.dirname(model_path) or "."
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_render_frame(env):
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    except Exception:
        frame = None
    return frame


def display_frames(frames, fps: float = 4.0, title: str | None = None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required to display animation frames")
        return

    if not frames:
        return

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")
    if title:
        fig.suptitle(title)

    def update(frame_index):
        im.set_data(frames[frame_index])
        return (im,)

    interval = 1000.0 / fps if fps > 0 else 250
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
    plt.show()
    return anim


def save_replay_frames(frames, model_path: str, episode: int):
    model_dir = os.path.dirname(model_path) or "."
    replay_dir = os.path.join(model_dir, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, f"episode_{episode:02d}.npz")
    np.savez_compressed(replay_path, frames=np.stack(frames))
    print(f"Saved replay frames to {replay_path}")
    return replay_path


def replay_saved_frames(args):
    model_dir = os.path.dirname(args.model_path) or "."
    replay_dir = os.path.join(model_dir, "replay")
    replay_files = sorted(glob.glob(os.path.join(replay_dir, "episode_*.npz")))
    if not replay_files:
        raise FileNotFoundError(f"No saved replay files found in {replay_dir}")
    episode_index = max(0, min(args.replay_episode - 1, len(replay_files) - 1))
    replay_file = replay_files[episode_index]
    data = np.load(replay_file)
    frames = data["frames"]
    display_frames(frames, fps=args.fps, title=os.path.basename(replay_file))


def evaluate(args):
    if args.replay:
        replay_saved_frames(args)
        return

    metadata = load_model_metadata(args.model_path)
    if metadata:
        saved_env = metadata.get("env_id")
        if args.env_id is None:
            args.env_id = saved_env
        elif saved_env is not None and args.env_id != saved_env:
            raise ValueError(
                f"Environment mismatch: model saved for {saved_env}, but --env-id is {args.env_id}. "
                "Use the same env_id as the saved model or omit --env-id to infer it."
            )
        if args.agent_type is None and metadata.get("method") in ["td0", "tdlambda", "tdlambda_actionfeatures", "tdlambda_cnn"]:
            args.agent_type = metadata.get("method")
        if metadata.get("obs_shape") is not None:
            args.obs_shape = tuple(metadata["obs_shape"])

    if args.agent_type is None:
        raise ValueError(
            "No agent type provided and no model metadata could infer it. "
            "Please pass --agent-type or save the model with a 'method' field in metadata."
        )

    if args.env_id is None:
        raise ValueError(
            "No env_id provided and no metadata found. Please pass --env-id to match the saved model."
        )

    env = make_env(
        args.env_id,
        seed=args.seed,
        use_manhattan_distance=args.use_manhattan_distance,
    )
    use_raw_observation = args.agent_type in {"tdlambda_actionfeatures", "tdlambda_cnn"}
    encoder = None if use_raw_observation else MiniGridEncoder(env.observation_space)
    if encoder is not None and metadata and metadata.get("state_size") != encoder.size:
        raise ValueError(
            f"Observation size mismatch: saved model expects state size {metadata.get('state_size')} "
            f"but env {args.env_id} produces state size {encoder.size}."
        )
    if use_raw_observation and metadata and "state_size" not in metadata:
        print(
            "Warning: saved model metadata does not contain state_size for tdlambda_actionfeatures. "
            "Evaluation will continue, but future saved models should include state_size."
        )
    if metadata and metadata.get("n_actions") != env.action_space.n:
        raise ValueError(
            f"Action space mismatch: saved model expects {metadata.get('n_actions')} actions "
            f"but env {args.env_id} has {env.action_space.n}."
        )

    agent = load_td_agent(
        args.agent_type,
        None if use_raw_observation else encoder.size,
        env.action_space.n,
        args.model_path,
        args,
    )

    results = []
    for episode in range(1, args.eval_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        if not use_raw_observation:
            state = encoder.encode(observation)
        terminated = False
        truncated = False
        total_reward = 0.0
        frames = []

        while not terminated and not truncated:
            if use_raw_observation:
                action = agent.greedy_action(observation)
            else:
                action = agent.greedy_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            if not use_raw_observation:
                state = encoder.encode(observation)
            total_reward += reward
            if args.render or args.save_replay:
                frame = get_render_frame(env)
                if frame is not None:
                    frames.append(frame)

        if args.render and frames:
            display_frames(frames, fps=args.fps, title=f"Episode {episode}")

        if args.save_replay and frames:
            save_replay_frames(frames, args.model_path, episode)

        results.append(total_reward)
        print(f"Eval episode {episode}: reward={total_reward:.3f}")

    mean_reward = float(np.mean(results))
    print(f"Average evaluation reward over {args.eval_episodes} episodes: {mean_reward:.3f}")

    save_evaluation_summary(results, args.model_path, args, metadata)
    save_evaluation_plot(results, args.model_path, args.env_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained TD agent")
    parser.add_argument("--env-id", type=str, default=None, help="Environment ID to use for evaluation. If omitted, the saved model metadata will be used.")
    parser.add_argument(
        "--agent-type",
        type=str,
        default=None,
        choices=["td0", "tdlambda", "tdlambda_actionfeatures", "tdlambda_cnn"],
        help="Agent type to evaluate. If omitted, the evaluator will infer it from the saved model metadata.",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render the episode as an animation")
    parser.add_argument("--save-replay", action="store_true", help="Save replay frames to the model folder for later playback")
    parser.add_argument("--replay", action="store_true", help="Replay a saved animation from the model folder")
    parser.add_argument("--replay-episode", type=int, default=1, help="Which saved episode replay to play")
    parser.add_argument("--fps", type=float, default=4.0, help="Playback frames per second for replay/render")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--lambda-value", type=float, default=0.8)
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=1,
        help="Radius of the local patch used by the action-conditional feature extractor.",
    )
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
