#!/usr/bin/env python3
"""
Batch generate GIF visualizations for all trained models in saved_models directory.
Calls visualize_agent.py for each model file (.zip for PPO, .npy for TD/Q).
GIFs are saved in the model's own directory with automatic naming.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
GIF_SCALE = 50      # Upscale factor for GIF frames (higher = larger, clearer)
GIF_FPS = 10        # Frames per second for output GIF
NO_DISPLAY = True   # Do not show GUI windows

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR
VISUALIZE_SCRIPT = PROJECT_ROOT / "visualize_agent.py"

if not VISUALIZE_SCRIPT.exists():
    print(f"Error: visualize_agent.py not found at {VISUALIZE_SCRIPT}")
    sys.exit(1)

def find_model_files(root_dir):
    """Yield (model_path, agent_type) for each model found."""
    root = Path(root_dir)
    # PPO models: .zip files (likely contain 'ppo' in name or path)
    for zip_file in root.glob("**/*.zip"):
        if "ppo" in str(zip_file).lower():
            yield zip_file, "ppo"
    # TD/Q models: .npy files (excluding episode_rewards.npy)
    for npy_file in root.glob("**/*.npy"):
        path_str = str(npy_file).lower()
        if "episode_rewards" in path_str:
            continue
        if "tdlambda" in path_str:
            yield npy_file, "tdlambda"
        elif "td0" in path_str:
            yield npy_file, "td0"
        elif "qlearning" in path_str:
            yield npy_file, "qlearning"
        else:
            print(f"Warning: unknown .npy file {npy_file}, skipping")

def run_visualization(model_path, agent_type):
    """Call visualize_agent.py for a single model."""
    cmd = [
        sys.executable,
        str(VISUALIZE_SCRIPT),
        "--agent-type", agent_type,
        "--model-path", str(model_path),
        "--save-gif",
        "--no-display",
        "--gif-scale", str(GIF_SCALE),
        "--gif-fps", str(GIF_FPS)
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"Success for {model_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed for {model_path}: {e}")
        return False

def main():
    models_root = Path("C:/Users/21934/RL_Project/saved_models")
    if not models_root.exists():
        print(f"Models directory not found: {models_root}")
        sys.exit(1)

    models = list(find_model_files(models_root))
    print(f"Found {len(models)} model files.")
    for p, t in models:
        print(f"  {t}: {p}")

    success = 0
    for model_path, agent_type in models:
        if run_visualization(model_path, agent_type):
            success += 1
    print(f"\nDone. {success}/{len(models)} succeeded.")

if __name__ == "__main__":
    main()