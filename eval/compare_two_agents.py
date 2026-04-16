#!/usr/bin/env python3
"""
Generic script to compare running mean curves of two RL algorithms.
Supported agents: ppo, tdlambda, qlearning.
Each agent must have a training script that saves episode_rewards.npy.
"""

'''
To run:
python eval/compare_two_agents.py --agent1 ppo --agent2 tdlambda --env-id Maze-Auto-Random \
    --agent1-args "--total-timesteps 50000" \
    --agent2-args "--lambda-value 0.9 --num-episodes 500" \
    --window 10

One row run:
python eval/compare_two_agents.py --agent1 ppo --agent2 tdlambda --env-id Maze-Auto-Random --agent1-args "--total-timesteps 50000" --agent2-args "--lambda-value 0.9 --num-episodes 500" --window 10
'''

import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "comparison_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
#  Agent configurations
# ----------------------------------------------------------------------
AGENT_CONFIG = {
    "ppo": {
        "train_script": "train_ppo.py",
        "required_args": ["--env-id", "--total-timesteps", "--seed"],
        "default_args": {
            "--total-timesteps": 10000,
            "--seed": 42
        },
        "extra_args": []
    },
    "tdlambda": {
        "train_script": "train_td.py",
        "required_args": ["--env-id", "--num-episodes", "--lambda-value", "--seed"],
        "default_args": {
            "--method": "tdlambda",
            "--alpha": 5e-4,
            "--gamma": 0.99,
            "--epsilon": 1.0,
            "--epsilon-decay": 0.995,
            "--epsilon-min": 0.05,
            "--seed": 42,
            "--num-episodes": 200,
            "--lambda-value": 0.8
        },
        "extra_args": []
    },
    "qlearning": {
        "train_script": "train_q.py",
        "required_args": ["--env-id", "--num-episodes", "--seed"],
        "default_args": {
            "--alpha": 5e-4,
            "--gamma": 0.99,
            "--epsilon": 1.0,
            "--epsilon-decay": 0.995,
            "--epsilon-min": 0.05,
            "--seed": 42,
            "--num-episodes": 200
        },
        "extra_args": []
    }
}

# ----------------------------------------------------------------------
#  Utility functions
# ----------------------------------------------------------------------
def run_training(agent_type, extra_args, save_dir_root="saved_models/comparison"):
    """Run training for given agent and return path to episode_rewards.npy."""
    config = AGENT_CONFIG[agent_type]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(PROJECT_ROOT, save_dir_root)
    os.makedirs(base_dir, exist_ok=True)
    outer_dir = os.path.join(base_dir, f"{timestamp}_{agent_type}")
    os.makedirs(outer_dir, exist_ok=True)

    # Build command
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "train", config["train_script"])] + extra_args + [
        "--save-dir", outer_dir
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Search for episode_rewards.npy
    for root, _, files in os.walk(outer_dir):
        if "episode_rewards.npy" in files:
            return root
    raise RuntimeError(f"Could not find episode_rewards.npy in {outer_dir}")

def load_rewards(run_dir):
    path = os.path.join(run_dir, "episode_rewards.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Episode rewards not found at {path}")
    return np.load(path)

def moving_average(data, window):
    if len(data) < window:
        return data, list(range(1, len(data)+1))
    kernel = np.ones(window)/window
    smoothed = np.convolve(data, kernel, mode='valid')
    x = range(window-1, len(data))
    return smoothed, x

def truncate_rewards(rewards, max_episodes):
    if len(rewards) > max_episodes:
        rewards = rewards[:max_episodes]
    return rewards

# ----------------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------------
def plot_running_mean_comparison(
    rewards1, label1, color1,
    rewards2, label2, color2,
    window, max_episodes=None,
    title=None, save_path=None
):
    if max_episodes is not None:
        rewards1 = truncate_rewards(rewards1, max_episodes)
        rewards2 = truncate_rewards(rewards2, max_episodes)

    smoothed1, x1 = moving_average(rewards1, window)
    smoothed2, x2 = moving_average(rewards2, window)

    plt.figure(figsize=(10, 5))
    plt.plot(x1, smoothed1, color=color1, linewidth=2, label=f"{label1} (running mean, w={window})")
    plt.plot(x2, smoothed2, color=color2, linewidth=2, label=f"{label2} (running mean, w={window})")

    if title is None:
        title = f"{label1} vs {label2} - Running Mean Comparison"
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare running mean curves of two RL agents")
    parser.add_argument("--agent1", type=str, required=True, choices=list(AGENT_CONFIG.keys()),
                        help="First agent type")
    parser.add_argument("--agent2", type=str, required=True, choices=list(AGENT_CONFIG.keys()),
                        help="Second agent type")
    parser.add_argument("--env-id", type=str, required=True, help="Environment ID")
    parser.add_argument("--window", type=int, default=10, help="Moving average window")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Truncate both reward sequences to this many episodes (for fair comparison)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: comparison_results/{agent1}_vs_{agent2}_{env-id}.png)")
    # Agent-specific override arguments (can be passed as --agent1-args "...")
    parser.add_argument("--agent1-args", type=str, default="",
                        help="Extra arguments for agent1 training (space separated)")
    parser.add_argument("--agent2-args", type=str, default="",
                        help="Extra arguments for agent2 training")
    parser.add_argument("--agent1-dir", type=str, help="Existing run directory for agent1 (skip training)")
    parser.add_argument("--agent2-dir", type=str, help="Existing run directory for agent2 (skip training)")
    parser.add_argument("--title", type=str, help="Custom plot title")
    args = parser.parse_args()

    if args.output is None:
        safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
        filename = f"{args.agent1}_vs_{args.agent2}_{safe_env_id}.png"
        args.output = os.path.join(RESULTS_DIR, filename)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get rewards for agent1
    if args.agent1_dir:
        rewards1 = load_rewards(args.agent1_dir)
        print(f"Loaded {args.agent1} rewards from {args.agent1_dir}")
    else:
        config = AGENT_CONFIG[args.agent1]
        extra = []
        # Add required base arguments
        if "--env-id" not in [a.split('=')[0] for a in extra] and "--env-id" not in args.agent1_args:
            extra.extend(["--env-id", args.env_id])
        # Add default args that are not overridden
        for k, v in config["default_args"].items():
            if k not in extra and k not in args.agent1_args:
                extra.extend([k, str(v)])
        # Add user-provided extra args
        if args.agent1_args:
            extra.extend(args.agent1_args.split())
        print(f"Training {args.agent1} with extra args: {extra}")
        run_dir = run_training(args.agent1, extra)
        rewards1 = load_rewards(run_dir)
        print(f"{args.agent1} training finished. Rewards saved in {run_dir}")

    # Get rewards for agent2
    if args.agent2_dir:
        rewards2 = load_rewards(args.agent2_dir)
        print(f"Loaded {args.agent2} rewards from {args.agent2_dir}")
    else:
        config = AGENT_CONFIG[args.agent2]
        extra = []
        if "--env-id" not in [a.split('=')[0] for a in extra] and "--env-id" not in args.agent2_args:
            extra.extend(["--env-id", args.env_id])
        for k, v in config["default_args"].items():
            if k not in extra and k not in args.agent2_args:
                extra.extend([k, str(v)])
        if args.agent2_args:
            extra.extend(args.agent2_args.split())
        print(f"Training {args.agent2} with extra args: {extra}")
        run_dir = run_training(args.agent2, extra)
        rewards2 = load_rewards(run_dir)
        print(f"{args.agent2} training finished. Rewards saved in {run_dir}")

    # Determine colors
    color_map = {
        "ppo": "green",
        "tdlambda": "orange",
        "qlearning": "blue"
    }
    color1 = color_map.get(args.agent1, "black")
    color2 = color_map.get(args.agent2, "black")

    # Plot
    plot_running_mean_comparison(
        rewards1, args.agent1.upper(), color1,
        rewards2, args.agent2.upper(), color2,
        window=args.window,
        max_episodes=args.max_episodes,
        title=args.title,
        save_path=args.output
    )

if __name__ == "__main__":
    main()