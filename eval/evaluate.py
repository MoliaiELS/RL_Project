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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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

def save_value_heatmap(agent, env, model_path, agent_type, episode_idx=None):
    """
    Generates and saves a V-Value (Max Q) heatmap to visualize the learned policy.
    Darker colors typically represent higher state values, showing the agent's preference.
    """
    if agent_type not in ["td0", "tdlambda", "tdlambda_actionfeatures"]:
        return
    
    if "MiniGrid" in str(type(env)) or not hasattr(env, "grid"):
        print(f"  [Info] Heatmap visualization is currently only optimized for custom MazeEnv. Skipping for MiniGrid.")
        return

    if agent_type not in ["td0", "tdlambda", "tdlambda_actionfeatures"]:
        return
    
    try:
        h = getattr(env, "height", None)
        w = getattr(env, "width", None)
        
        if h is None or w is None:
            return

        v_table = np.zeros((h, w))

        encoder = None if agent_type == "tdlambda_actionfeatures" else MiniGridEncoder(env.observation_space)
        # Iterate through every cell in the grid to calculate its maximum state value
        for y in range(h):
            for x in range(w):
                if env.grid[y, x] == '#':
                    v_table[y, x] = -0.5  # Walls are assigned a low constant value for visualization
                    continue
                
                # Temporarily set agent position to probe the local state value
                env.agent_pos = (y, x)
                obs = env._get_observation()
                
                if agent_type == "tdlambda_actionfeatures":
                    q_vals = agent.q_values(obs)
                else:
                    q_vals = agent.q_values(encoder.encode(obs))
                
                v_table[y, x] = np.max(q_vals)

        plt.figure(figsize=(8, 6))
        sns.heatmap(v_table, annot=False, cmap="YlGnBu")
        if episode_idx is not None:
            title = f"Value Function Heatmap ({agent_type}) - Ep {episode_idx}"
            filename = f"policy_heatmap_ep{episode_idx:02d}.png"
        else:
            title = f"Value Function Heatmap ({agent_type})"
            filename = "policy_heatmap.png"
        
        model_dir = os.path.dirname(model_path)
        heatmap_path = os.path.join(model_dir, filename)
        plt.savefig(heatmap_path)
        plt.close()
        print(f"  [Success] Saved policy heatmap to {heatmap_path}")

    except Exception as e:
        print(f"  [Warning] Could not generate heatmap: {e}")

def save_replay_frames(frames, model_path, episode):
    """Saves a sequence of frames as a compressed .npz file for GIF conversion."""
    model_dir = os.path.dirname(model_path)
    replay_dir = os.path.join(model_dir, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, f"episode_{episode:02d}.npz")
    np.savez_compressed(replay_path, frames=np.stack(frames))
    print(f"  [Video] Saved replay to {replay_path}")

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
        # obs_shape = getattr(args, "obs_shape", None)
        obs_shape = getattr(args, "obs_shape", (8, 8, 3)) if obs_shape is None else tuple(obs_shape)
        if obs_shape is None:
            raise ValueError("obs_shape metadata is required to load tdlambda_cnn models.")
        agent = TDLambdaCNNAgent(
            obs_shape=tuple(obs_shape),
            n_actions=n_actions,
            gamma=args.gamma,
            alpha=args.alpha,
            epsilon=0.0,
            seed=args.seed,
            device=args.device,
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

def evaluate_single_model(model_path, args):
    """
    Standardized evaluation logic for a single weight file.
    Outputs: Success Rate, Avg Steps, Heatmaps, and JSON summaries.
    """
    print(f"\n>>> Processing: {model_path}")
    
    # Attempt to load metadata from the model's directory
    model_dir = os.path.dirname(model_path)
    meta_path = os.path.join(model_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    # Fallback to metadata if arguments are not explicitly provided
    env_id = args.env_id or metadata.get("env_id", "Maze-Easy")
    agent_type = args.agent_type or metadata.get("method", "tdlambda")
    
    args.gamma = metadata.get("gamma", args.gamma)
    args.alpha = metadata.get("alpha", args.alpha)
    args.lambda_value = metadata.get("lambda_value", args.lambda_value)
    args.patch_radius = metadata.get("patch_radius", args.patch_radius)
    
    # Environment Setup
    env = make_env(env_id, seed=args.seed, use_manhattan_distance=args.use_manhattan_distance)
    use_raw_obs = agent_type in ["tdlambda_actionfeatures", "tdlambda_cnn"]
    encoder = None if use_raw_obs else MiniGridEncoder(env.observation_space)
    
    # Agent Setup and Weight Loading
    agent = load_td_agent(agent_type, None if use_raw_obs else encoder.size, env.action_space.n, model_path, args)

    results = []
    steps_record = []
    success_record = []

    # Execution of Evaluation Episodes
    for episode in range(1, args.eval_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        state = None if use_raw_obs else encoder.encode(observation)
        
        total_reward = 0.0
        steps = 0
        done = False
        frames = []
        
        while not done and steps < 200:
            if args.save_replay:
                try:
                    frame = get_render_frame(env)
                    if frame is not None:
                        frames.append(frame)
                except Exception:
                        pass
                    
            # Deterministic/Greedy Action Selection
            action = agent.greedy_action(observation if use_raw_obs else state)
            observation, reward, terminated, truncated, _ = env.step(action)
            if not use_raw_obs:
                state = encoder.encode(observation)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated

        results.append(total_reward)
        steps_record.append(steps)
        # Definition of success: Reached goal with positive cumulative reward
        success_record.append(1 if (done and reward > 0 and steps < 200) else 0)

        if "Random" in env_id: 
            save_value_heatmap(agent, env, model_path, agent_type, episode_idx=episode)
        
        # 2. Handle Fixed Maze Heatmap (Only save once on the first episode to save time)
        elif episode == 1:
            save_value_heatmap(agent, env, model_path, agent_type)
        
        if args.save_replay and frames:
            save_replay_frames(frames, model_path, episode)

    # Results Consolidation
    mean_reward = np.mean(results)
    success_rate = np.mean(success_record)
    avg_steps = np.mean(steps_record)

    print(f"\nAverage evaluation reward: {mean_reward:.3f} | Success: {success_rate:.1%} | Avg Steps: {avg_steps:.1f},")

    # Generate visual and structured data deliverables
    save_value_heatmap(agent, env, model_path, agent_type)
    
    summary = {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "mean_reward": mean_reward,
        "episodes": args.eval_episodes
    }
    with open(os.path.join(model_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Integrated Evaluation and Batch Analysis Suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a single .npy file OR a directory of models")
    parser.add_argument("--env-id", type=str, default=None, help="Target Environment ID")
    parser.add_argument("--agent-type", type=str, default=None, help="Agent type (td0, tdlambda, etc.)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument("--save-replay", action="store_true", help="Capture and save video frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument("--use-manhattan-distance", action="store_true", default=True)
    parser.add_argument("--patch-radius", type=int, default=1, help="Radius for action features")
    args = parser.parse_args()

    # Determine execution mode: Batch vs Single Model
    if os.path.isdir(args.model_path):
        print(f"Batch Mode: Scanning for models in {args.model_path}")
        # Search for weight files while filtering out reward history files
        all_files = glob.glob(os.path.join(args.model_path, "**", "*.npy"), recursive=True)
        model_files = [f for f in all_files if "reward" not in os.path.basename(f).lower()]
        
        if not model_files:
            print("No valid weight files (.npy) found in the directory.")
            return

        for m_path in model_files:
            try:
                evaluate_single_model(m_path, args)
            except Exception as e:
                print(f"Skipping {m_path} due to error: {e}")
    else:
        # Evaluate a specific single file
        evaluate_single_model(args.model_path, args)

if __name__ == "__main__":
    main()