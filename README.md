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

## Quick start

Train TD(λ) on a MiniGrid task:

```bash
python train/train_td.py --env-id MiniGrid-Empty-8x8-v0 --num-episodes 200 --lambda-value 0.8
```

Train PPO for comparison:

```bash
python train/train_ppo.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000
```

Evaluate a trained TD agent:

```bash
python eval/evaluate.py --env-id MiniGrid-Empty-8x8-v0 --model-path saved_td_lambda.npy
```

## Notes
- `TDZeroAgent` uses linear Q-learning with tile-style state encoding.
- `TDLambdaAgent` uses linear Q(λ) with accumulating eligibility traces.
- PPO is implemented using `stable-baselines3` for policy-based comparison.
