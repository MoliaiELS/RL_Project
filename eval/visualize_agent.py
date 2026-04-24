#!/usr/bin/env python3
"""
Universal script to visualize a trained agent (PPO, TD(0), TD(λ), Q-learning)
and optionally save the episode as a GIF.

Usage examples:
  # Visualize PPO (auto-reads metadata)
  python visualize_agent.py --agent-type ppo --model-path path/to/model.zip

  # Visualize TD(λ) and save GIF (metadata used automatically)
  python visualize_agent.py --agent-type tdlambda --model-path path/to/tdlambda.npy --save-gif

  # Override env-id manually if metadata missing
  python visualize_agent.py --agent-type td0 --model-path path/to/td0.npy --env-id Maze-Easy --save-gif
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.minigrid_env import make_env, MiniGridEncoder
from agents.td_zero import TDZeroAgent
from agents.td_lambda import TDLambdaAgent
from agents.q_learning import QLearningAgent

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. PPO visualization disabled.")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. GIF saving disabled. Install with: pip install imageio")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. GIF will not be upscaled. Install with: pip install Pillow")


def load_metadata(model_path):
    """Load metadata.json from the same directory as the model file."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    meta_path = os.path.join(model_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained agent (PPO, TD(0), TD(λ), Q-learning)")
    parser.add_argument("--agent-type", type=str, required=True,
                        choices=["ppo", "td0", "tdlambda", "qlearning"],
                        help="Type of agent to visualize")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model file (.zip for PPO, .npy for TD/Q)")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Environment ID (overrides metadata if provided)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment")
    parser.add_argument("--fps", type=float, default=4.0,
                        help="Frames per second for rendering (controls delay when showing)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                        help="Use stochastic actions (only for PPO)")
    parser.add_argument("--save-gif", type=str, nargs='?', const="__auto__", default=None,
                        help="Save episode as GIF. Optionally provide filename. If no filename, auto-generate as MODELNAME_ENVID.gif in model directory.")
    parser.add_argument("--gif-fps", type=float, default=10.0,
                        help="FPS for the output GIF (default: 10)")
    parser.add_argument("--gif-scale", type=int, default=10,
                        help="Upscale factor for GIF frames (default: 10, set to 1 to disable upscaling)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not display the window, only save GIF (if --save-gif provided)")
    # Override hyperparameters (if metadata not available)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Learning rate (used if metadata missing)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (used if metadata missing)")
    parser.add_argument("--lambda-value", type=float, default=None,
                        help="Lambda for TD(λ) (used if metadata missing)")
    return parser.parse_args()


def load_agent(args, metadata):
    """Load agent based on type, model path, and metadata."""
    env_id = args.env_id if args.env_id else metadata.get("env_id", "Maze-Easy")
    use_manhattan = metadata.get("use_manhattan_distance", True)
    
    # For PPO, we only need the environment to be consistent; no extra params
    if args.agent_type == "ppo":
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for PPO")
        # Create environment with same parameters as training (if available)
        use_manhattan = metadata.get("use_manhattan_distance", True)
        env = make_env(env_id, seed=args.seed, use_manhattan_distance=use_manhattan)
        # Load model
        model = PPO.load(args.model_path, env=env)
        env.close()
        class PPOAgentWrapper:
            def __init__(self, model):
                self.model = model
            def predict(self, obs, deterministic=True):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                return action
        return PPOAgentWrapper(model), env_id, use_manhattan
    
    # For TD and Q-learning, we need to create agent with proper hyperparameters
    alpha = args.alpha if args.alpha is not None else metadata.get("alpha", 5e-4)
    gamma = args.gamma if args.gamma is not None else metadata.get("gamma", 0.99)
    lambda_val = args.lambda_value if args.lambda_value is not None else metadata.get("lambda_value", 0.9)
    
    use_manhattan = metadata.get("use_manhattan_distance", True)
    temp_env = make_env(env_id, seed=args.seed, use_manhattan_distance=use_manhattan)
    encoder = MiniGridEncoder(temp_env.observation_space)
    state_size = encoder.size
    n_actions = temp_env.action_space.n
    temp_env.close()
    
    if args.agent_type == "td0":
        agent = TDZeroAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=gamma,
            alpha=alpha,
            epsilon=0.0,
            seed=args.seed
        )
    elif args.agent_type == "tdlambda":
        agent = TDLambdaAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=gamma,
            alpha=alpha,
            epsilon=0.0,
            lambda_value=lambda_val,
            seed=args.seed
        )
    elif args.agent_type == "qlearning":
        agent = QLearningAgent(
            state_size=state_size,
            n_actions=n_actions,
            gamma=gamma,
            alpha=alpha,
            epsilon=0.0,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unsupported agent type: {args.agent_type}")
    
    agent.load(args.model_path)
    class TDWrapper:
        def __init__(self, agent, encoder):
            self.agent = agent
            self.encoder = encoder
        def predict(self, obs, deterministic=True):
            state = self.encoder.encode(obs)
            if deterministic:
                action = self.agent.greedy_action(state)
            else:
                action = self.agent.select_action(state)
            return action
    return TDWrapper(agent, encoder), env_id, use_manhattan


def main():
    args = parse_args()
    
    metadata = load_metadata(args.model_path)
    
    try:
        agent, env_id, use_manhattan = load_agent(args, metadata)
        print(f"Loaded {args.agent_type} agent from {args.model_path}")
        print(f"Using environment: {env_id} (use_manhattan_distance={use_manhattan})")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        sys.exit(1)
    
    # Handle GIF save path
    if args.save_gif is not None:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
        if args.save_gif == "__auto__":
            args.save_gif = f"{model_basename}_{env_id}.gif"
        if not os.path.isabs(args.save_gif):
            args.save_gif = os.path.join(model_dir, args.save_gif)
        os.makedirs(os.path.dirname(args.save_gif), exist_ok=True)
    
    env = make_env(env_id, seed=args.seed, use_manhattan_distance=use_manhattan)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    frames = []
    
    if not args.no_display:
        plt.ion()
        fig, ax = plt.subplots()
        ax.axis("off")
        im = None
    
    print(f"Starting visualization on {env_id}...")
    try:
        while not done:
            action = agent.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            frame = env.render(mode="rgb_array")
            if frame is None:
                print("Warning: render returned None, skipping frame")
                continue
            frames.append(frame)
            
            if not args.no_display:
                if im is None:
                    im = ax.imshow(frame)
                else:
                    im.set_data(frame)
                plt.draw()
                plt.pause(1.0 / args.fps)
        
        print(f"Episode finished in {step} steps, total reward: {total_reward:.3f}")
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        if args.save_gif is not None and HAS_IMAGEIO and frames:
            scale = args.gif_scale
            if scale > 1 and HAS_PIL:
                print(f"Upscaling frames by factor {scale} for better quality...")
                upscaled_frames = []
                for frame in frames:
                    pil_img = Image.fromarray(frame)
                    new_size = (pil_img.width * scale, pil_img.height * scale)
                    pil_img = pil_img.resize(new_size, Image.NEAREST)
                    upscaled_frames.append(np.array(pil_img))
                frames_to_save = upscaled_frames
            else:
                frames_to_save = frames
                if scale > 1 and not HAS_PIL:
                    print("Warning: Pillow not installed, using original low resolution GIF.")
            print(f"Saving GIF to {args.save_gif} with {len(frames_to_save)} frames...")
            duration = 1.0 / args.gif_fps
            try:
                imageio.mimsave(args.save_gif, frames_to_save, format='GIF', duration=duration, loop=0)
                print(f"GIF saved to {args.save_gif}")
            except Exception as e:
                print(f"Failed to save GIF: {e}")
        elif args.save_gif and not HAS_IMAGEIO:
            print("Cannot save GIF: imageio not installed.")
        
        if not args.no_display:
            plt.close('all')
            plt.ioff()
        env.close()
        if not args.no_display:
            print("Visualization ended. You can close the window.")


if __name__ == "__main__":
    main()