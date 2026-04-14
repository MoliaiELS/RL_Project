import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_ENVS = ["Maze-Easy", "Maze-Auto-9x9", "MiniGrid-Empty-8x8-v0"]
DEFAULT_METHOD = "tdlambda"


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value):
    return [float(item) for item in parse_list(value)]


def parse_ints(value):
    return [int(item) for item in parse_list(value)]


def build_command(env_id, method, alpha, gamma, lambda_value, epsilon, epsilon_decay, epsilon_min, num_episodes, seed, save_dir):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "train", "train_td.py"),
        "--env-id",
        env_id,
        "--method",
        method,
        "--alpha",
        str(alpha),
        "--gamma",
        str(gamma),
        "--lambda-value",
        str(lambda_value),
        "--epsilon",
        str(epsilon),
        "--epsilon-decay",
        str(epsilon_decay),
        "--epsilon-min",
        str(epsilon_min),
        "--num-episodes",
        str(num_episodes),
        "--seed",
        str(seed),
        "--save-dir",
        save_dir,
    ]
    return cmd


def run_command(cmd, cwd):
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return process.returncode, process.stdout, process.stderr


def write_summary(rows, output_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Explore training hyperparameters for TD(0)/TD(\u03bb) on Maze and MiniGrid environments"
    )
    parser.add_argument(
        "--env-ids",
        type=str,
        default=",".join(DEFAULT_ENVS),
        help="Comma-separated env IDs to explore. Defaults to Maze-Easy,Maze-Auto-9x9,MiniGrid-Empty-8x8-v0",
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=["td0", "tdlambda"])
    parser.add_argument("--alphas", type=parse_floats, default=[5e-4, 1e-3], help="Comma-separated alpha values")
    parser.add_argument("--gammas", type=parse_floats, default=[0.95, 0.99], help="Comma-separated gamma values")
    parser.add_argument("--lambda-values", type=parse_floats, default=[0.7, 0.8, 0.9], help="Comma-separated lambda values for TD(lambda)")
    parser.add_argument("--epsilons", type=parse_floats, default=[1.0], help="Comma-separated starting epsilon values")
    parser.add_argument("--epsilon-decays", type=parse_floats, default=[0.995, 0.99], help="Comma-separated epsilon decay values")
    parser.add_argument("--epsilon-mins", type=parse_floats, default=[0.05], help="Comma-separated epsilon min values")
    parser.add_argument("--num-episodes", type=parse_ints, default=[100], help="Comma-separated numbers of training episodes")
    parser.add_argument("--seeds", type=parse_ints, default=[42], help="Comma-separated random seeds")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_models"), help="Base directory to save experiment runs")
    parser.add_argument("--results-file", type=str, default=os.path.join(PROJECT_ROOT, "saved_models", "param_search_results.csv"), help="CSV file where summary results are written")
    parser.add_argument("--eval-after-train", action="store_true", help="Run evaluation after each training job and record average reward")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes when --eval-after-train is enabled")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()

    env_ids = parse_list(args.env_ids)
    table = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.abspath(args.output_dir)
    os.makedirs(result_dir, exist_ok=True)

    for env_id in env_ids:
        for alpha, gamma, lambda_value, epsilon, epsilon_decay, epsilon_min, num_episodes, seed in itertools.product(
            args.alphas,
            args.gammas,
            args.lambda_values,
            args.epsilons,
            args.epsilon_decays,
            args.epsilon_mins,
            args.num_episodes,
            args.seeds,
        ):
            run_dir_name = f"paramsearch-{env_id.replace('/', '_')}-{args.method}-a{alpha}-g{gamma}-l{lambda_value}-e{epsilon}-d{epsilon_decay}-m{epsilon_min}-n{num_episodes}-s{seed}-{timestamp}"
            run_save_dir = os.path.join(result_dir, run_dir_name)
            os.makedirs(run_save_dir, exist_ok=True)

            cmd = build_command(
                env_id=env_id,
                method=args.method,
                alpha=alpha,
                gamma=gamma,
                lambda_value=lambda_value,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                num_episodes=num_episodes,
                seed=seed,
                save_dir=run_save_dir,
            )

            row = {
                "env_id": env_id,
                "method": args.method,
                "alpha": alpha,
                "gamma": gamma,
                "lambda_value": lambda_value,
                "epsilon": epsilon,
                "epsilon_decay": epsilon_decay,
                "epsilon_min": epsilon_min,
                "num_episodes": num_episodes,
                "seed": seed,
                "save_dir": run_save_dir,
                "train_returncode": None,
                "train_stdout": None,
                "train_stderr": None,
                "eval_returncode": None,
                "eval_mean_reward": None,
                "error": None,
            }

            print(f"Running experiment: {env_id}, alpha={alpha}, gamma={gamma}, lambda={lambda_value}, eps={epsilon}, decay={epsilon_decay}, eps_min={epsilon_min}, episodes={num_episodes}, seed={seed}")
            if args.dry_run:
                print(" ".join(cmd))
                table.append(row)
                continue

            returncode, stdout, stderr = run_command(cmd, cwd=PROJECT_ROOT)
            row["train_returncode"] = returncode
            row["train_stdout"] = stdout.strip()
            row["train_stderr"] = stderr.strip()
            if returncode != 0:
                row["error"] = "Training failed"
                print(f"  Training failed: return code {returncode}")
                print(stderr)
                table.append(row)
                continue

            if args.eval_after_train:
                trained_model_glob = os.path.join(run_save_dir, f"{args.method}_{env_id.replace('/', '_')}.npy")
                eval_cmd = [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "eval", "evaluate.py"),
                    "--model-path",
                    trained_model_glob,
                    "--eval-episodes",
                    str(args.eval_episodes),
                ]
                if env_id.startswith("Maze-"):
                    eval_cmd.extend(["--env-id", env_id])
                print(f"  Evaluating trained model: {trained_model_glob}")
                eval_returncode, eval_stdout, eval_stderr = run_command(eval_cmd, cwd=PROJECT_ROOT)
                row["eval_returncode"] = eval_returncode
                if eval_returncode != 0:
                    row["error"] = "Evaluation failed"
                    row["train_stderr"] += "\n" + eval_stderr.strip()
                    print(f"  Evaluation failed: return code {eval_returncode}")
                    print(eval_stderr)
                else:
                    for line in eval_stdout.splitlines():
                        if "Average evaluation reward" in line:
                            try:
                                row["eval_mean_reward"] = float(line.split(":")[-1].strip())
                            except ValueError:
                                pass

            table.append(row)

    write_summary(table, args.results_file)
    print(f"Wrote parameter search summary to {args.results_file}")


if __name__ == "__main__":
    main()
