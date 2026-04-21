from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train", "train_td_graph_gpu.py")


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return value
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_command(cmd: list[str], cwd: str) -> None:
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


def build_stage_configs(args) -> list[dict]:
    stage_episodes = parse_int_list(args.stage_episodes)
    if len(stage_episodes) != 6:
        raise ValueError("--stage-episodes must contain exactly 6 values")

    stage_epsilons = parse_float_list(args.stage_epsilons)
    if len(stage_epsilons) != 6:
        raise ValueError("--stage-epsilons must contain exactly 6 values")

    stage_epsilon_decays = parse_float_list(args.stage_epsilon_decays)
    if len(stage_epsilon_decays) != 6:
        raise ValueError("--stage-epsilon-decays must contain exactly 6 values")

    random_ratios = parse_float_list(args.stage_random_ratios)
    if len(random_ratios) != 4:
        raise ValueError("--stage-random-ratios must contain exactly 4 values for stages 2-5")

    if args.stage_learning_rates is None:
        stage_learning_rates = [
            args.alpha,
            args.alpha,
            args.alpha,
            args.alpha * 0.75,
            args.alpha * 0.75,
            args.alpha * 0.5,
        ]
    else:
        stage_learning_rates = parse_float_list(args.stage_learning_rates)
        if len(stage_learning_rates) != 6:
            raise ValueError("--stage-learning-rates must contain exactly 6 values")

    if args.pretrain_seeds <= 0:
        raise ValueError("--pretrain-seeds must be positive")

    pretrain_seeds = [args.seed + idx for idx in range(args.pretrain_seeds)]

    def mixed_probs(random_ratio: float) -> list[float]:
        fixed_ratio = max(0.0, 1.0 - random_ratio)
        return [fixed_ratio / 3.0] * 3 + [random_ratio]

    return [
        {
            "stage_name": "stage_01_pretrain_fixed_random_seeds",
            "save_name": "stage_01_pretrain_fixed_random_seeds",
            "env_ids": ["Maze-Auto-Random-9x9"],
            "env_probs": [1.0],
            "fixed_seeds": pretrain_seeds,
            "seed_probs": [1.0 / len(pretrain_seeds)] * len(pretrain_seeds),
            "num_episodes": stage_episodes[0],
            "epsilon": stage_epsilons[0],
            "epsilon_decay": stage_epsilon_decays[0],
            "epsilon_hold_episodes": stage_episodes[0],
            "learning_rate": stage_learning_rates[0],
        },
        {
            "stage_name": "stage_02_mix_random_10pct",
            "save_name": "stage_02_mix_random_10pct",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": mixed_probs(random_ratios[0]),
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": stage_episodes[1],
            "epsilon": stage_epsilons[1],
            "epsilon_decay": stage_epsilon_decays[1],
            "epsilon_hold_episodes": min(args.stage_hold_episodes, stage_episodes[1]),
            "learning_rate": stage_learning_rates[1],
        },
        {
            "stage_name": "stage_03_mix_random_20pct",
            "save_name": "stage_03_mix_random_20pct",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": mixed_probs(random_ratios[1]),
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": stage_episodes[2],
            "epsilon": stage_epsilons[2],
            "epsilon_decay": stage_epsilon_decays[2],
            "epsilon_hold_episodes": min(args.stage_hold_episodes, stage_episodes[2]),
            "learning_rate": stage_learning_rates[2],
        },
        {
            "stage_name": "stage_04_mix_random_40pct",
            "save_name": "stage_04_mix_random_40pct",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": mixed_probs(random_ratios[2]),
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": stage_episodes[3],
            "epsilon": stage_epsilons[3],
            "epsilon_decay": stage_epsilon_decays[3],
            "epsilon_hold_episodes": min(args.stage_hold_episodes, stage_episodes[3]),
            "learning_rate": stage_learning_rates[3],
        },
        {
            "stage_name": "stage_05_mix_random_70pct",
            "save_name": "stage_05_mix_random_70pct",
            "env_ids": ["Maze-Easy", "Maze-Medium", "Maze-Hard", "Maze-Auto-Random-9x9"],
            "env_probs": mixed_probs(random_ratios[3]),
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": stage_episodes[4],
            "epsilon": stage_epsilons[4],
            "epsilon_decay": stage_epsilon_decays[4],
            "epsilon_hold_episodes": min(args.stage_hold_episodes, stage_episodes[4]),
            "learning_rate": stage_learning_rates[4],
        },
        {
            "stage_name": "stage_06_full_random",
            "save_name": "stage_06_full_random",
            "env_ids": ["Maze-Auto-Random-9x9"],
            "env_probs": [1.0],
            "fixed_seeds": None,
            "seed_probs": None,
            "num_episodes": stage_episodes[5],
            "epsilon": stage_epsilons[5],
            "epsilon_decay": stage_epsilon_decays[5],
            "epsilon_hold_episodes": min(args.stage_hold_episodes, stage_episodes[5]),
            "learning_rate": stage_learning_rates[5],
        },
    ]


def build_cmd(args, stage_config: dict, previous_model_path: str | None) -> tuple[list[str], str]:
    stage_dir = os.path.join(args.save_dir, args.curriculum_name, stage_config["save_name"])
    cmd = [
        sys.executable,
        "-u",
        TRAIN_SCRIPT,
        "--env-ids",
        ",".join(stage_config["env_ids"]),
        "--env-probs",
        ",".join(str(prob) for prob in stage_config["env_probs"]),
        "--num-episodes",
        str(stage_config["num_episodes"]),
        "--alpha",
        str(stage_config["learning_rate"]),
        "--gamma",
        str(args.gamma),
        "--epsilon",
        str(stage_config["epsilon"]),
        "--epsilon-min",
        str(args.epsilon_min),
        "--epsilon-decay",
        str(stage_config["epsilon_decay"]),
        "--epsilon-hold-episodes",
        str(stage_config["epsilon_hold_episodes"]),
        "--lambda-value",
        str(args.lambda_value),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--gradient-clip",
        str(args.gradient_clip),
        "--target-tau",
        str(args.target_tau),
        "--target-update-freq",
        str(args.target_update_freq),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-episodes",
        str(args.eval_episodes),
        "--log-interval",
        str(args.log_interval),
        "--save-name",
        stage_config["save_name"],
        "--stage-name",
        stage_config["stage_name"],
        "--run-dir",
        stage_dir,
        "--save-dir",
        args.save_dir,
        "--seed",
        str(args.seed),
    ]

    if stage_config.get("fixed_seeds"):
        cmd.extend(["--fixed-seeds", ",".join(str(seed) for seed in stage_config["fixed_seeds"])])
    if stage_config.get("seed_probs"):
        cmd.extend(["--seed-probs", ",".join(str(prob) for prob in stage_config["seed_probs"])])
    if args.device:
        cmd.extend(["--device", args.device])
    if not args.use_manhattan_distance:
        cmd.append("--no-manhattan-distance")
    if args.use_target_network:
        cmd.append("--use-target-network")
    else:
        cmd.append("--no-target-network")
    if previous_model_path is not None:
        cmd.extend(["--load-model", previous_model_path])

    return cmd, stage_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum training for graph-based TD(lambda) maze agent")
    parser.add_argument("--curriculum-name", type=str, default="graph_td_curriculum")
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-min", type=float, default=0.08)
    parser.add_argument("--lambda-value", type=float, default=0.7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-manhattan-distance",
        action="store_false",
        dest="use_manhattan_distance",
        default=True,
        help="Disable Manhattan distance reward shaping for Maze environments.",
    )
    parser.add_argument(
        "--use-target-network",
        action="store_true",
        dest="use_target_network",
        default=True,
        help="Use a target network for bootstrap stabilization.",
    )
    parser.add_argument(
        "--no-target-network",
        action="store_false",
        dest="use_target_network",
        help="Disable the target network.",
    )
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-update-freq", type=int, default=1)
    parser.add_argument(
        "--stage-episodes",
        type=str,
        default="3000,4000,5000,6000,7000,9000",
        help="Comma-separated episode counts for the 6 curriculum stages.",
    )
    parser.add_argument(
        "--stage-random-ratios",
        type=str,
        default="0.1,0.2,0.4,0.7",
        help="Comma-separated random-maze ratios for stages 2-5.",
    )
    parser.add_argument(
        "--stage-epsilons",
        type=str,
        default="0.30,0.28,0.24,0.20,0.16,0.12",
        help="Comma-separated epsilon starts for the 6 curriculum stages.",
    )
    parser.add_argument(
        "--stage-epsilon-decays",
        type=str,
        default="1.0,0.99995,0.99995,0.99996,0.99997,0.99998",
        help="Comma-separated epsilon decay values for the 6 curriculum stages.",
    )
    parser.add_argument(
        "--stage-learning-rates",
        type=str,
        default=None,
        help="Comma-separated learning rates for the 6 curriculum stages.",
    )
    parser.add_argument(
        "--stage-hold-episodes",
        type=int,
        default=1000,
        help="How long to keep epsilon fixed before stage-level decay starts.",
    )
    parser.add_argument(
        "--pretrain-seeds",
        type=int,
        default=3,
        help="Number of fixed random maze seeds used in stage 1.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    curriculum_root = os.path.join(args.save_dir, args.curriculum_name)
    os.makedirs(curriculum_root, exist_ok=True)

    stage_configs = build_stage_configs(args)
    previous_model_path = None
    curriculum_summary = {
        "curriculum_name": args.curriculum_name,
        "stages": [],
    }

    for stage_config in stage_configs:
        cmd, run_dir = build_cmd(args, stage_config, previous_model_path)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Running stage: {stage_config['stage_name']}")
        try:
            run_command(cmd, cwd=PROJECT_ROOT)
        except RuntimeError as exc:
            print(f"Stage {stage_config['stage_name']} failed: {exc}")
            sys.exit(1)

        previous_model_path = os.path.join(run_dir, f"{stage_config['save_name']}.pt")
        stage_summary_path = os.path.join(run_dir, "stage_summary.json")
        stage_metrics_path = os.path.join(run_dir, "metrics.json")
        stage_eval_path = os.path.join(run_dir, "greedy_eval_history.json")

        stage_summary = {
            "stage_name": stage_config["stage_name"],
            "run_dir": run_dir,
            "env_ids": stage_config["env_ids"],
            "env_probs": stage_config["env_probs"],
            "fixed_seeds": stage_config["fixed_seeds"],
            "num_episodes": stage_config["num_episodes"],
            "epsilon": stage_config["epsilon"],
            "epsilon_decay": stage_config["epsilon_decay"],
            "epsilon_hold_episodes": stage_config["epsilon_hold_episodes"],
            "learning_rate": stage_config["learning_rate"],
            "model_path": previous_model_path,
            "metrics_path": stage_metrics_path,
            "eval_path": stage_eval_path,
        }

        if os.path.isfile(stage_summary_path):
            with open(stage_summary_path, "r", encoding="utf-8") as summary_file:
                stage_summary["result"] = json.load(summary_file)

        curriculum_summary["stages"].append(stage_summary)

    summary_path = os.path.join(curriculum_root, "curriculum_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(curriculum_summary, summary_file, indent=2)
    print(f"Saved curriculum summary to {summary_path}")


if __name__ == "__main__":
    main()
