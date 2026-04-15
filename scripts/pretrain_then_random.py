import argparse
import glob
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train", "train_td.py")

# Reference configuration from the successful fixed-maze TD(lambda) run.
REFERENCE_PRETRAIN_CONFIG = {
    "num_episodes": 10000,
    "alpha": 5e-4,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.9995,
    "epsilon_min": 0.0,
    "lambda_value": 0.8,
    "eval_interval": 100,
    "eval_episodes": 5,
}

REFERENCE_FINETUNE_CONFIG = {
    "alpha": 3e-4,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.9997,
    "epsilon_min": 0.10,
    "lambda_value": 0.8,
    "eval_interval": 100,
    "eval_episodes": 5,
}


def run_command(cmd, cwd):
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return process.returncode, process.stdout, process.stderr


def build_train_cmd(env_id, model_path=None, **kwargs):
    cmd = [sys.executable, TRAIN_SCRIPT, "--env-id", env_id, "--method", "tdlambda"]
    for key, value in kwargs.items():
        if value is None:
            continue
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))
    if model_path:
        cmd.extend(["--load-model", model_path])
    return cmd


def parse_seed_list(seed_text: str | None, base_seed: int) -> list[int]:
    if not seed_text:
        return [base_seed + offset for offset in range(4)]
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("pretrain seed list must not be empty")
    return seeds


def split_episode_budget(total_episodes: int, num_stages: int) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("total pretrain episodes must be positive")
    if num_stages <= 0:
        raise ValueError("number of curriculum stages must be positive")
    base = total_episodes // num_stages
    remainder = total_episodes % num_stages
    allocation = [base + (1 if i < remainder else 0) for i in range(num_stages)]
    return [max(1, value) for value in allocation]


def find_saved_model_path(base_dir: str, env_id: str, method: str = "tdlambda") -> str | None:
    metadata_files = sorted(
        glob.glob(os.path.join(base_dir, "**", "metadata.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not metadata_files:
        return None
    for meta_path in reversed(metadata_files):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("env_id") == env_id and meta.get("method") == method:
            model_dir = meta.get("save_dir")
            if model_dir and os.path.isdir(model_dir):
                model_files = glob.glob(os.path.join(model_dir, f"{method}_{env_id.replace('/', '_')}.npy"))
                if model_files:
                    return model_files[0]
    return None


def ensure_model_path_exists(model_path: str, env_id: str, save_dir: str) -> str:
    if model_path and os.path.isfile(model_path):
        return model_path
    discovered = find_saved_model_path(save_dir, env_id)
    if discovered is None:
        raise FileNotFoundError(f"Unable to locate trained model for {env_id} inside {save_dir}")
    return discovered


def run_training_stage(
    *,
    stage_name: str,
    env_id: str,
    save_dir: str,
    num_episodes: int,
    seed: int,
    model_path: str | None = None,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    lambda_value: float,
    eval_interval: int,
    eval_episodes: int,
):
    os.makedirs(save_dir, exist_ok=True)
    cmd = build_train_cmd(
        env_id,
        model_path=model_path,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        lambda_value=lambda_value,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        seed=seed,
        save_dir=save_dir,
    )
    print(f"[{stage_name}] Running:", " ".join(cmd))
    returncode, stdout, stderr = run_command(cmd, cwd=PROJECT_ROOT)
    if returncode != 0:
        print(f"[{stage_name}] Training failed")
        if stdout:
            print(stdout)
        print(stderr)
        sys.exit(returncode)
    trained_model_path = ensure_model_path_exists(None, env_id, save_dir)
    return {
        "stage_name": stage_name,
        "env_id": env_id,
        "save_dir": save_dir,
        "num_episodes": num_episodes,
        "seed": seed,
        "load_model": model_path,
        "trained_model_path": trained_model_path,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "lambda_value": lambda_value,
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training for TD(lambda): multiple fixed mazes followed by random-maze fine-tuning"
    )
    parser.add_argument("--fixed-env", type=str, default="Maze-Auto-9x9")
    parser.add_argument("--random-env", type=str, default="Maze-Auto-Random-9x9")
    parser.add_argument(
        "--pretrain-episodes",
        type=int,
        default=REFERENCE_PRETRAIN_CONFIG["num_episodes"],
        help="Total pretraining episode budget, split across the fixed-maze seed curriculum.",
    )
    parser.add_argument(
        "--pretrain-seeds",
        type=str,
        default=None,
        help="Comma-separated fixed-maze seeds. Defaults to seed, seed+1, seed+2, seed+3.",
    )
    parser.add_argument("--finetune-episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=REFERENCE_PRETRAIN_CONFIG["alpha"])
    parser.add_argument("--gamma", type=float, default=REFERENCE_PRETRAIN_CONFIG["gamma"])
    parser.add_argument("--epsilon", type=float, default=REFERENCE_PRETRAIN_CONFIG["epsilon"])
    parser.add_argument("--epsilon-decay", type=float, default=REFERENCE_PRETRAIN_CONFIG["epsilon_decay"])
    parser.add_argument("--epsilon-min", type=float, default=REFERENCE_PRETRAIN_CONFIG["epsilon_min"])
    parser.add_argument("--lambda-value", type=float, default=REFERENCE_PRETRAIN_CONFIG["lambda_value"])
    parser.add_argument("--eval-interval", type=int, default=REFERENCE_PRETRAIN_CONFIG["eval_interval"])
    parser.add_argument("--eval-episodes", type=int, default=REFERENCE_PRETRAIN_CONFIG["eval_episodes"])
    parser.add_argument("--finetune-alpha", type=float, default=REFERENCE_FINETUNE_CONFIG["alpha"])
    parser.add_argument("--finetune-gamma", type=float, default=REFERENCE_FINETUNE_CONFIG["gamma"])
    parser.add_argument("--finetune-epsilon", type=float, default=REFERENCE_FINETUNE_CONFIG["epsilon"])
    parser.add_argument("--finetune-epsilon-decay", type=float, default=REFERENCE_FINETUNE_CONFIG["epsilon_decay"])
    parser.add_argument("--finetune-epsilon-min", type=float, default=REFERENCE_FINETUNE_CONFIG["epsilon_min"])
    parser.add_argument("--finetune-lambda-value", type=float, default=REFERENCE_FINETUNE_CONFIG["lambda_value"])
    parser.add_argument("--finetune-eval-interval", type=int, default=REFERENCE_FINETUNE_CONFIG["eval_interval"])
    parser.add_argument("--finetune-eval-episodes", type=int, default=REFERENCE_FINETUNE_CONFIG["eval_episodes"])
    parser.add_argument("--fixed-model-path", type=str, default=None, help="Use an existing pretrained fixed-maze model instead of training one from scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_models"))
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    curriculum_summary = {
        "fixed_env": args.fixed_env,
        "random_env": args.random_env,
        "base_seed": args.seed,
        "pretrain_total_episodes": args.pretrain_episodes,
        "finetune_episodes": args.finetune_episodes,
        "stages": [],
    }

    fixed_model_path = args.fixed_model_path
    if fixed_model_path is None:
        pretrain_root_dir = os.path.join(
            args.save_dir,
            f"pretrain-{args.fixed_env}-{args.pretrain_episodes}-{args.seed}",
        )
        os.makedirs(pretrain_root_dir, exist_ok=True)

        pretrain_seeds = parse_seed_list(args.pretrain_seeds, args.seed)
        pretrain_episode_allocation = split_episode_budget(args.pretrain_episodes, len(pretrain_seeds))
        current_model_path = None

        print("Fixed-maze curriculum seeds:", pretrain_seeds)
        print("Episode allocation per seed:", pretrain_episode_allocation)

        for stage_index, (stage_seed, stage_episodes) in enumerate(
            zip(pretrain_seeds, pretrain_episode_allocation),
            start=1,
        ):
            stage_save_dir = os.path.join(
                pretrain_root_dir,
                f"stage{stage_index:02d}-seed{stage_seed}",
            )
            stage_summary = run_training_stage(
                stage_name=f"pretrain-stage-{stage_index}",
                env_id=args.fixed_env,
                save_dir=stage_save_dir,
                num_episodes=stage_episodes,
                seed=stage_seed,
                model_path=current_model_path,
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_decay=args.epsilon_decay,
                epsilon_min=args.epsilon_min,
                lambda_value=args.lambda_value,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
            )
            curriculum_summary["stages"].append(stage_summary)
            current_model_path = stage_summary["trained_model_path"]

        fixed_model_path = current_model_path
        if fixed_model_path is None:
            raise FileNotFoundError("Unable to locate pretrained model after fixed-maze curriculum")
    else:
        if not os.path.isfile(fixed_model_path):
            raise FileNotFoundError(f"Specified fixed model path does not exist: {fixed_model_path}")
        curriculum_summary["stages"].append(
            {
                "stage_name": "external-pretrained-model",
                "env_id": args.fixed_env,
                "trained_model_path": fixed_model_path,
            }
        )

    random_run_dir = os.path.join(
        args.save_dir,
        f"finetune-{args.random_env}-{args.finetune_episodes}-{args.seed}",
    )
    finetune_summary = run_training_stage(
        stage_name="random-finetune",
        env_id=args.random_env,
        save_dir=random_run_dir,
        num_episodes=args.finetune_episodes,
        seed=args.seed,
        model_path=fixed_model_path,
        alpha=args.finetune_alpha,
        gamma=args.finetune_gamma,
        epsilon=args.finetune_epsilon,
        epsilon_decay=args.finetune_epsilon_decay,
        epsilon_min=args.finetune_epsilon_min,
        lambda_value=args.finetune_lambda_value,
        eval_interval=args.finetune_eval_interval,
        eval_episodes=args.finetune_eval_episodes,
    )
    curriculum_summary["stages"].append(finetune_summary)

    summary_path = os.path.join(random_run_dir, "curriculum_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(curriculum_summary, f, indent=2)

    print(f"Pretraining and fine-tuning complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
