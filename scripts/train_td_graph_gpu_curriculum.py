import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train", "train_td_graph_gpu.py")


def run_command(cmd, cwd):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess output")

    for line in process.stdout:
        print(line, end="")

    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


def build_cmd(args, stage_config, previous_model_path: str | None):
    cmd = [sys.executable, "-u", TRAIN_SCRIPT]
    env_ids = stage_config["env_ids"]
    env_probs = stage_config["env_probs"]
    cmd += ["--env-ids", ",".join(env_ids)]
    cmd += ["--env-probs", ",".join(str(p) for p in env_probs)]
    if stage_config.get("fixed_seeds"):
        cmd += ["--fixed-seeds", ",".join(str(seed) for seed in stage_config["fixed_seeds"])]
    if stage_config.get("seed_probs"):
        cmd += ["--seed-probs", ",".join(str(p) for p in stage_config["seed_probs"])]
    cmd += ["--num-episodes", str(stage_config["num_episodes"])]
    cmd += ["--alpha", str(args.alpha)]
    cmd += ["--gamma", str(args.gamma)]
    cmd += ["--epsilon", str(args.epsilon)]
    cmd += ["--epsilon-min", str(args.epsilon_min)]
    # Use more stable epsilon decay for curriculum training
    if args.stable_transition:
        cmd += ["--epsilon-decay", "0.9998"]  # Slower decay
    else:
        cmd += ["--epsilon-decay", str(args.epsilon_decay)]
    cmd += ["--lambda-value", str(args.lambda_value)]
    cmd += ["--hidden-dim", str(args.hidden_dim)]
    cmd += ["--num-layers", str(args.num_layers)]
    cmd += ["--dropout", str(args.dropout)]
    if not args.use_manhattan_distance:
        cmd.append("--no-manhattan-distance")
    cmd += ["--eval-interval", str(args.eval_interval)]
    cmd += ["--eval-episodes", str(args.eval_episodes)]
    cmd += ["--log-interval", str(args.log_interval)]
    cmd += ["--save-name", stage_config["save_name"]]
    stage_dir = os.path.join(args.save_dir, args.curriculum_name, stage_config["save_name"])
    cmd += ["--run-dir", stage_dir]
    if args.device:
        cmd += ["--device", args.device]
    if stage_config.get("epsilon") is not None:
        cmd += ["--epsilon", str(stage_config["epsilon"])]
    if previous_model_path is not None:
        cmd += ["--load-model", previous_model_path]
    cmd += ["--seed", str(args.seed)]
    return cmd, stage_dir


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated curriculum training for graph-based TD(lambda) maze agent")
    parser.add_argument("--curriculum-name", type=str, default="graph_td_gpu_curriculum")
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.08)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument(
        "--stage-epsilons",
        type=lambda v: [float(item.strip()) for item in v.split(",") if item.strip()],
        default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        help="Comma-separated starting epsilon values for each curriculum stage.",
    )
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda", help="Device for GPU acceleration (default: cuda)")
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument(
        "--stage-episodes",
        type=lambda v: [int(item.strip()) for item in v.split(",") if item.strip()],
        default=[800, 1000, 1500, 2000, 1500, 2000],
        help="Comma-separated number of episodes for each stage in the curriculum.",
    )
    parser.add_argument(
        "--stage-random-ratios",
        type=lambda v: [float(item.strip()) for item in v.split(",") if item.strip()],
        default=[0.05, 0.15, 0.3],
        help="Comma-separated random-maze ratios for stage 3, 4 and 5.",
    )
    parser.add_argument(
        "--pretrain-seeds",
        type=int,
        default=3,
        help="Number of fixed random seeds to use during the first pretraining stage.",
    )
    parser.add_argument(
        "--stable-transition",
        action="store_true",
        default=True,
        help="Use stable transition mode: slower epsilon decay and learning rate adjustment.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    curriculum_root = os.path.join(args.save_dir, args.curriculum_name)
    os.makedirs(curriculum_root, exist_ok=True)

    p_3maze = ["Maze-Auto-Random-9x9"]
    p_3maze_seeds = [args.seed + i for i in range(args.pretrain_seeds)]
    p_rand = ["Maze-Auto-Random-9x9"]

    if len(args.stage_episodes) != 6:
        raise ValueError("--stage-episodes must contain exactly 6 values")
    if len(args.stage_random_ratios) != 3:
        raise ValueError("--stage-random-ratios must contain exactly 3 values for stages 3,4,5")
    if len(args.stage_epsilons) != 6:
        raise ValueError("--stage-epsilons must contain exactly 6 values")

    stage_configs = [
        {
            "save_name": "stage_01_pretrain_3maps",
            "env_ids": p_3maze,
            "env_probs": [1.0],
            "fixed_seeds": p_3maze_seeds,
            "seed_probs": [1.0] * len(p_3maze_seeds),
            "num_episodes": args.stage_episodes[0],
            "epsilon": args.stage_epsilons[0],
        },
        {
            "save_name": "stage_02_fixed_difficulty",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard"],
            "env_probs": [1.0/3.0] * 3,
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": args.stage_episodes[1],
            "epsilon": args.stage_epsilons[1],
        },
        {
            "save_name": "stage_03_mix_random_low",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": [((1.0 - args.stage_random_ratios[0]) / 3)] * 3 + [args.stage_random_ratios[0]],
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": args.stage_episodes[2],
            "epsilon": args.stage_epsilons[2],
        },
        {
            "save_name": "stage_04_mix_random_med",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": [((1.0 - args.stage_random_ratios[1]) / 3)] * 3 + [args.stage_random_ratios[1]],
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": args.stage_episodes[3],
            "epsilon": args.stage_epsilons[3],
        },
        {
            "save_name": "stage_05_mix_random_high",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": [((1.0 - args.stage_random_ratios[2]) / 3)] * 3 + [args.stage_random_ratios[2]],
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": args.stage_episodes[4],
            "epsilon": args.stage_epsilons[4],
        },
        {
            "save_name": "stage_06_full_random",
            "env_ids": ["Maze-Auto-Random-9x9"],
            "env_probs": [1.0],
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": args.stage_episodes[5],
            "epsilon": args.stage_epsilons[5],
        },
    ]

    previous_model_path = None
    summary = []
    for stage_config in stage_configs:
        cmd, run_dir = build_cmd(args, stage_config, previous_model_path)
        print("Running stage:", stage_config["save_name"])
        os.makedirs(run_dir, exist_ok=True)
        try:
            run_command(cmd, cwd=PROJECT_ROOT)
        except RuntimeError as exc:
            print(f"Stage {stage_config['save_name']} failed: {exc}")
            sys.exit(1)

        previous_model_path = os.path.join(run_dir, f"{stage_config['save_name']}.pt")
        stage_summary = {
            "stage": stage_config["save_name"],
            "run_dir": run_dir,
            "env_ids": stage_config["env_ids"],
            "env_probs": stage_config["env_probs"],
            "fixed_seeds": stage_config["fixed_seeds"],
            "num_episodes": stage_config["num_episodes"],
            "model_path": previous_model_path,
        }
        summary.append(stage_summary)

    summary_path = os.path.join(curriculum_root, "curriculum_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved curriculum summary to {summary_path}")


if __name__ == "__main__":
    main()