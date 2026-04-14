# Robot Navigation with TD(λ)

This project implements a MiniGrid-based robot navigation benchmark and compares TD(0), TD(λ), and PPO.

## Structure
- `envs/`: MiniGrid environment creation and state encoding
- `agents/`: TD(0), TD(λ) and PPO agent implementations
- `train/`: training entrypoints for TD algorithms and PPO
- `eval/`: evaluation utilities
- `utils/`: plotting and helper utilities

## Setup
1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you still have the deprecated `gym-minigrid` package installed, replace it with `minigrid`:

```bash
pip uninstall gym-minigrid
pip install minigrid
```

The project also supports auto-generated mazes via `mazelib` when installed. If `mazelib` is not available, the code falls back to a built-in maze generator.

## Quick start

Train TD(λ) on a maze task with obstacles:

```bash
python train/train_td.py --env-id Maze-Easy --num-episodes 200 --lambda-value 0.8
```

Train TD(λ) on a harder staged maze:

```bash
python train/train_td.py --env-id Maze-Stage --num-episodes 200 --lambda-value 0.8
```

Train TD(λ) on an auto-generated maze:

```bash
python train/train_td.py --env-id Maze-Auto-9x9 --num-episodes 200 --lambda-value 0.8
```

Train TD(λ) on a MiniGrid task:

```bash
python train/train_td.py --env-id MiniGrid-Empty-8x8-v0 --num-episodes 200 --lambda-value 0.8 --epsilon 1.0 --epsilon-decay 0.995 --epsilon-min 0.05
```

Train PPO for comparison:

```bash
python train/train_ppo.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000
```

Model and plot files are saved under a timestamped folder inside `saved_models` by default, for example:

```text
saved_models/20260414-133548-tdlambda-Maze-Easy/
  tdlambda_Maze-Easy.npy
  tdlambda_learning_curve.png
```

Evaluate a trained TD agent and save replay frames for later playback. Evaluation now also writes `eval_summary.json` and `eval_rewards.png` into the same model folder:

```bash
python eval/evaluate.py --env-id MiniGrid-Empty-8x8-v0 --model-path saved_models/<timestamp>-tdlambda-MiniGrid-Empty-8x8-v0/tdlambda_MiniGrid-Empty-8x8-v0.npy --render --save-replay
```

Replay a saved TD agent animation from the model folder:

```bash
python eval/evaluate.py --env-id MiniGrid-Empty-8x8-v0 --model-path saved_models/<timestamp>-tdlambda-MiniGrid-Empty-8x8-v0/tdlambda_MiniGrid-Empty-8x8-v0.npy --replay
```

If the saved model contains metadata, you can omit `--env-id` and the evaluator will infer the environment from the model folder:

```bash
python eval/evaluate.py --model-path saved_models/<timestamp>-tdlambda-MiniGrid-Empty-8x8-v0/tdlambda_MiniGrid-Empty-8x8-v0.npy --replay
```

Render a trained TD agent in the maze env:

```bash
python eval/evaluate.py --env-id Maze-Easy --model-path saved_models/<timestamp>-tdlambda-Maze-Easy/tdlambda_Maze-Easy.npy --render --save-replay
```

Replay a saved maze replay:

```bash
python eval/evaluate.py --env-id Maze-Easy --model-path saved_models/<timestamp>-tdlambda-Maze-Easy/tdlambda_Maze-Easy.npy --replay
```

## Notes
- `TDZeroAgent` uses linear Q-learning with tile-style state encoding.
- `TDLambdaAgent` is implemented from scratch using linear Q(λ) with accumulating eligibility traces.
- PPO is implemented using `stable-baselines3` for policy-based comparison.
