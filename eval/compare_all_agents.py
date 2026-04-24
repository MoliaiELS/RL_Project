#!/usr/bin/env python3
"""
Compare multiple agents (TD0, TDλ, Qlearning, PPO, TDλ+ActionFeatures) by training them from scratch.
All training runs are saved under saved_models/comparison/<timestamp>_<env-id>/,
and a combined running mean plot is generated.

Usage examples:
  # Compare all five agents on Maze-Stage with 3000 episodes / 50000 steps for PPO
  python compare_all_agents.py --env-id Maze-Stage --agents td0 tdlambda qlearning ppo tdlambda_actionfeatures \\
      --window 30 --agent-args td0 "--num-episodes 3000" tdlambda "--num-episodes 3000 --lambda-value 0.9" \\
      qlearning "--num-episodes 3000" ppo "--total-timesteps 50000" tdlambda_actionfeatures "--num-episodes 3000 --lambda-value 0.9"
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# ----------------------------------------------------------------------
# Agent configurations
# ----------------------------------------------------------------------
AGENT_CONFIG = {
    "ppo": {
        "train_script": "train_ppo.py",
        "required_args": ["--env-id", "--total-timesteps", "--seed"],
        "default_args": {
            "--total-timesteps": 8000,   # roughly 200 episodes for Maze-Easy (40 steps/ep)
            "--seed": 42
        },
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
    },
    "td0": {
        "train_script": "train_td.py",
        "required_args": ["--env-id", "--num-episodes", "--seed"],
        "default_args": {
            "--method": "td0",
            "--alpha": 5e-4,
            "--gamma": 0.99,
            "--epsilon": 1.0,
            "--epsilon-decay": 0.995,
            "--epsilon-min": 0.05,
            "--seed": 42,
            "--num-episodes": 200
        },
    },
    "tdlambda_actionfeatures": {
        "train_script": "train_td_action_features.py",
        "required_args": ["--env-id", "--num-episodes", "--seed"],
        "default_args": {
            "--alpha": 5e-4,
            "--gamma": 0.99,
            "--epsilon": 1.0,
            "--epsilon-decay": 0.995,
            "--epsilon-min": 0.05,
            "--seed": 42,
            "--num-episodes": 200,
            "--lambda-value": 0.9,
            "--patch-radius": 1,
        },
    }
}

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def run_training(agent_type, extra_args, base_save_dir):
    """Train an agent and return the directory containing episode_rewards.npy."""
    config = AGENT_CONFIG[agent_type]
    # Create a subdirectory for this agent inside the base save directory
    agent_dir = os.path.join(base_save_dir, agent_type)
    os.makedirs(agent_dir, exist_ok=True)

    # Build command
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "train", config["train_script"])] + extra_args + [
        "--save-dir", agent_dir
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Search for episode_rewards.npy (should be in a timestamp subfolder inside agent_dir)
    for root, _, files in os.walk(agent_dir):
        if "episode_rewards.npy" in files:
            return root
    raise RuntimeError(f"Could not find episode_rewards.npy for {agent_type} in {agent_dir}")

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
    if max_episodes is not None and len(rewards) > max_episodes:
        return rewards[:max_episodes]
    return rewards

# ----------------------------------------------------------------------
# Parsing of --agent-args
# ----------------------------------------------------------------------
def parse_agent_args(agent_args_list):
    """
    Parse --agent-args arguments in the form: agent_name "args_string" agent_name "args_string" ...
    Returns a dict {agent_name: list_of_args_strings_split}
    """
    agent_args = {}
    it = iter(agent_args_list)
    for agent in it:
        if agent in AGENT_CONFIG:
            args_str = next(it, None)
            if args_str is None:
                raise ValueError(f"Missing argument string for agent {agent}")
            agent_args[agent] = args_str.split()
        else:
            raise ValueError(f"Unknown agent name in --agent-args: {agent}")
    return agent_args

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple RL agents")
    parser.add_argument("--env-id", type=str, required=True, help="Environment ID")
    parser.add_argument("--agents", type=str, nargs='+', required=True,
                        choices=list(AGENT_CONFIG.keys()),
                        help="List of agent types to compare (e.g., td0 tdlambda qlearning ppo tdlambda_actionfeatures)")
    parser.add_argument("--agent-args", type=str, nargs='*', default=[],
                        help="Agent-specific arguments in the form: agent_name \"args\" agent_name \"args\" ...")
    parser.add_argument("--window", type=int, default=10, help="Moving average window")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Truncate all reward sequences to this many episodes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: inside the experiment folder)")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    args = parser.parse_args()

    # Create base experiment folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env_id = args.env_id.replace('/', '_').replace(' ', '_')
    exp_dir = os.path.join(PROJECT_ROOT, "saved_models", "comparison", f"{timestamp}_{safe_env_id}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # Parse agent-specific arguments
    agent_extra_args = parse_agent_args(args.agent_args) if args.agent_args else {}

    # Train each agent and collect rewards
    rewards_dict = {}
    for agent in args.agents:
        if agent in agent_extra_args:
            extra = agent_extra_args[agent]
        else:
            extra = []
        # Add required env-id if not present
        if "--env-id" not in extra and f"--env-id={args.env_id}" not in extra:
            extra.extend(["--env-id", args.env_id])
        # Add default args that are not overridden
        default_args = AGENT_CONFIG[agent]["default_args"]
        for k, v in default_args.items():
            # Check if already present (as full argument or as key=value)
            if k not in extra and not any(arg.startswith(k) for arg in extra):
                extra.extend([k, str(v)])
        print(f"\n--- Training {agent} ---")
        run_dir = run_training(agent, extra, exp_dir)
        rewards = load_rewards(run_dir)
        rewards_dict[agent] = rewards
        print(f"Loaded {len(rewards)} episodes for {agent}")

    # Determine common episode length
    min_len = min(len(r) for r in rewards_dict.values())
    if args.max_episodes is not None:
        min_len = min(min_len, args.max_episodes)
    for agent in rewards_dict:
        rewards_dict[agent] = truncate_rewards(rewards_dict[agent], min_len)

    # Color map
    color_map = {
        "ppo": "green",
        "tdlambda": "orange",
        "qlearning": "blue",
        "td0": "red",
        "tdlambda_actionfeatures": "purple"
    }

    # Plot
    plt.figure(figsize=(12, 6))
    for agent, rewards in rewards_dict.items():
        smoothed, x = moving_average(rewards, args.window)
        color = color_map.get(agent, "black")
        plt.plot(x, smoothed, linewidth=2, label=f"{agent.upper()} (running mean, w={args.window})", color=color)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    if args.title:
        plt.title(args.title)
    else:
        plt.title(f"Agent Comparison on {args.env_id}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if args.output is None:
        output_path = os.path.join(exp_dir, f"comparison_{safe_env_id}.png")
    else:
        output_path = args.output
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()