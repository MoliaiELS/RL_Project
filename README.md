# Robot Navigation with TD Control Variants

This repository implements a small reinforcement learning benchmark for robot navigation in grid mazes and MiniGrid environments. The codebase centers on temporal-difference control, starting from linear TD/Q baselines and extending toward richer state representations for navigation:

- `TD(0)` and linear `TD(lambda)` control on flattened observations
- `TD(lambda)` with action-conditional handcrafted navigation features
- `TD(lambda)` with a CNN over raw maze observations
- `TD(lambda)` with a graph encoder for topology-aware generalization
- comparison baselines with linear Q-learning and PPO

The project also includes custom maze environments, evaluation/replay tools for several agents, and curriculum-style scripts for dynamic and randomized maze training.

## What Is In This Repo

- `envs/`
  Custom maze environments and MiniGrid environment wrappers.
- `agents/`
  Agent implementations for Q-learning, TD variants, CNN TD, graph TD, and PPO.
- `train/`
  Main training entrypoints for the supported algorithms.
- `scripts/`
  Higher-level experiment scripts, including dynamic maze scheduling and graph curriculum training.
- `eval/`
  Evaluation, replay, and baseline comparison utilities.
- `utils/`
  Plotting helpers used by training and evaluation scripts.

## Environment Support

The repository supports two environment families:

- `Maze-*`
  Custom 3-channel maze observations with wall, goal, and agent planes.
- `MiniGrid-*`
  Standard MiniGrid environments wrapped with `FlatObsWrapper`.

Built-in maze IDs include:

- `Maze-Easy`
- `Maze-Medium`
- `Maze-Hard`
- `Maze-Stage`
- `Maze-Auto`
- `Maze-Auto-Random`
- `Maze-Auto-9x9`
- `Maze-Auto-Random-9x9`

Notes:

- `Maze-Auto-*` uses generated mazes. If `mazelib` is unavailable, the code falls back to an internal maze generator.
- Maze environments use shaped rewards by default. You can disable Manhattan-distance shaping with `--no-manhattan-distance`.
- `train_td.py` supports both `Maze-*` and `MiniGrid-*`.
- `train_td_action_features.py`, `train_td_cnn.py`, and `train_td_graph.py` are maze-focused and expect the custom 3-channel maze observation format.

## Algorithms

### 1. Linear TD control

`train/train_td.py` supports:

- `td0`
  Linear TD/Q-style control on flattened observations.
- `tdlambda`
  Linear `TD(lambda)` control with eligibility traces and on-policy next-action bootstrapping.

This is the simplest control pipeline in the repo and is the best starting point if you want a compact baseline.

### 2. TD(lambda) with action-conditional features

`train/train_td_action_features.py` replaces flat state encoding with explicit per-action navigation features such as:

- invalid move indicator
- distance-to-goal change
- free-neighbor ratio
- dead-end and corridor heuristics
- local wall and goal patches around the candidate next position

This keeps the TD control framework simple while making the value function more navigation-aware.

### 3. TD(lambda) with a CNN

`train/train_td_cnn.py` learns action values directly from raw maze observations using a small convolutional network and eligibility traces in parameter space.

### 4. TD(lambda) with a graph encoder

`train/train_td_graph.py` converts the maze into a graph of traversable cells, encodes the graph with message passing, and predicts action values from local topology-aware embeddings. This is the most general representation in the repo and is the main direction for cross-maze generalization.

### 5. Additional baselines

- `train/train_q.py`
  Linear Q-learning baseline.
- `train/train_ppo.py`
  PPO baseline built on `stable-baselines3`.

## Setup

### Option 1: existing Conda environment

If you already use the course/project Conda environment:

```powershell
& "D:\programfile\anaconda3\shell\condabin\conda-hook.ps1"
conda activate RL_Project
```

### Option 2: fresh Python environment

```bash
pip install -r requirements.txt
```

Main dependencies:

- `gymnasium`
- `minigrid`
- `mazelib`
- `numpy`
- `torch`
- `matplotlib`
- `stable-baselines3`

If you still have the older `gym-minigrid` package installed, prefer the modern `minigrid` package instead.

## Quick Start

### Linear TD(lambda) on a fixed maze

```bash
python -m train.train_td --env-id Maze-Easy --method tdlambda --num-episodes 300
```

### TD(0) baseline

```bash
python -m train.train_td --env-id Maze-Easy --method td0 --num-episodes 300
```

### Linear TD(lambda) on MiniGrid

```bash
python -m train.train_td --env-id MiniGrid-Empty-8x8-v0 --method tdlambda --num-episodes 500
```

### Action-feature TD(lambda)

```bash
python -m train.train_td_action_features --env-id Maze-Easy --num-episodes 300 --patch-radius 2
```

### CNN TD(lambda)

```bash
python -m train.train_td_cnn --env-id Maze-Easy --num-episodes 300 --device cpu
```

### Graph TD(lambda)

```bash
python -m train.train_td_graph --env-id Maze-Auto-Random-9x9 --num-episodes 800 --device cpu
```

### Q-learning baseline

```bash
python -m train.train_q --env-id Maze-Easy --num-episodes 250
```

### PPO baseline

```bash
python -m train.train_ppo --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000
```

## Advanced Training Scripts

### Dynamic maze scheduling for action-feature TD

`scripts/train_td_action_features_dynamic_maze.py` supports a two-phase setup:

- optional pretraining on a fixed auto-generated maze
- later training on randomized mazes with configurable change frequency

Example:

```bash
python -m scripts.train_td_action_features_dynamic_maze \
  --pretrain-env-id Maze-Auto \
  --pretrain-episodes 400 \
  --env-id Maze-Auto-Random-9x9 \
  --num-episodes 1200 \
  --stage-lengths 300,400,500 \
  --change-frequencies 50,20,1
```

This is useful when you want to move from stable early learning to progressively harder generalization.

### Curriculum training for graph TD

`scripts/train_td_graph_curriculum.py` runs a multi-stage curriculum that moves from fixed random seeds toward fully randomized mazes.

Example:

```bash
python -m scripts.train_td_graph_curriculum \
  --curriculum-name graph_td_curriculum \
  --stage-episodes 400,300,300,300,500 \
  --stage-random-ratios 0.2,0.5,0.8 \
  --device cpu
```

Each stage launches `train/train_td_graph.py`, saves its own run directory, and passes the previous stage checkpoint into the next stage.

## Evaluation and Replay

`eval/evaluate.py` currently supports evaluation for:

- `td0`
- `tdlambda`
- `tdlambda_actionfeatures`
- `tdlambda_cnn`

If `metadata.json` is present, the script can infer the saved environment and agent type automatically.

### Evaluate a saved model

```bash
python -m eval.evaluate --model-path saved_models/<run_dir>/tdlambda_Maze-Easy.npy
```

### Render evaluation episodes

```bash
python -m eval.evaluate --model-path saved_models/<run_dir>/tdlambda_Maze-Easy.npy --render
```

### Save replay frames

```bash
python -m eval.evaluate --model-path saved_models/<run_dir>/tdlambda_Maze-Easy.npy --save-replay
```

### Replay saved frames later

```bash
python -m eval.evaluate --model-path saved_models/<run_dir>/tdlambda_Maze-Easy.npy --replay
```

Evaluation writes:

- `eval_summary.json`
- `eval_rewards.png`
- optional `replay/episode_XX.npz`

Note:

- the current evaluator does not yet load `tdlambda_graph`, `qlearning`, or `ppo` checkpoints.

## Comparing Two Baselines

`eval/compare_two_agents.py` can train and compare two agents by plotting running-mean reward curves.

Currently supported in that script:

- `ppo`
- `tdlambda`
- `qlearning`

Example:

```bash
python -m eval.compare_two_agents \
  --agent1 ppo \
  --agent2 tdlambda \
  --env-id Maze-Auto-Random-9x9 \
  --agent1-args "--total-timesteps 50000" \
  --agent2-args "--num-episodes 500 --lambda-value 0.9" \
  --window 10
```

## Compare Multiple Agents

`eval/compare_all_agents.py` trains three or more agents in one run and produces a single plot with all running‑mean curves.

Supported agents: `td0`, `tdlambda`, `qlearning`, `ppo`, `tdlambda_actionfeatures`.

```bash
python -m eval.compare_all_agents \
  --env-id Maze-Stage \
  --agents td0 tdlambda qlearning ppo tdlambda_actionfeatures \
  --window 30 \
  --agent-args td0 "--num-episodes 3000" \
               tdlambda "--num-episodes 3000 --lambda-value 0.9" \
               qlearning "--num-episodes 3000" \
               ppo "--total-timesteps 50000" \
               tdlambda_actionfeatures "--num-episodes 3000 --lambda value 0.9"
```

The combined plot is stored inside `saved_models/comparison/<timestamp>_<env-id>/comparison_<env-id>.png.`

## Visualizing a Trained Agent
`eval/visualize_agent.py` runs one greedy episode of any trained agent and can optionally save a GIF or display the episode live.

Supported agent types: `ppo`, `td0`, `tdlambda`, `qlearning`, `tdlambda_actionfeatures`.

**Generate a GIF (no window)**

```bash
python -m eval.visualize_agent \
  --agent-type tdlambda \
  --model-path saved_models/.../tdlambda_Maze-Easy.npy \
  --save-gif --no-display
```

**Show the episode in a window**

```bash
python -m eval.visualize_agent \
  --agent-type ppo \
  --model-path saved_models/.../ppo_Maze-Easy.zip \
  --env-id Maze-Easy --render
```

**Key options:**

`--save-gif` : save the episode as a GIF (auto‑named <model_basename>_<env-id>.gif).

`--gif-scale` : upscaling factor for the GIF (default 10) – improves visibility in presentations.

`--no-display` : suppress the live matplotlib window (useful for batch generation).

`--env-id` : environment ID; if omitted, read from metadata.json.

**Requirements**: imageio and Pillow (pip install imageio Pillow).

## Batch GIF Generation

`eval/batch_generate_gifs.py` scans all models in `saved_models/` and automatically generates a GIF for each one.

Supports PPO (`.zip`) and linear agents (`.npy` with names `tdlambda_*`, `td0_*`, `qlearning_*`).

Configuration (hardcoded at the top of the script):` GIF_SCALE = 10`, `GIF_FPS = 10`, `NO_DISPLAY = True`.

```bash
python -m eval.batch_generate_gifs
Each GIF is saved next to its original model file as <model_basename>_<env-id>.gif.
```

## Saved Outputs

Training runs are saved into timestamped directories under `saved_models/` by default. A typical run contains some or all of the following:

- trained model checkpoint
- `metadata.json`
- `episode_rewards.npy`
- learning-curve plot
- combined training/evaluation plot
- `greedy_eval_history.json`

Example:

```text
saved_models/20260422-122229-tdlambda-Maze-Easy/
  metadata.json
  episode_rewards.npy
  tdlambda_Maze-Easy.npy
  tdlambda_learning_curve.png
```

## Important Current Limitations

- `action_features`, `cnn`, and `graph` training are designed for the custom maze observation format, not generic MiniGrid observations.
- `eval/evaluate.py` does not yet support graph checkpoints.
- `eval/compare_two_agents.py` only compares `ppo`, `tdlambda`, and `qlearning`.
- Some older files still contain encoding artifacts in comments or strings, but the main training paths are intact.

## Recommended Starting Points

If you are new to the repo:

1. Start with `train/train_td.py` on `Maze-Easy`.
2. Move to `train/train_td_action_features.py` to see how representation changes improve navigation behavior.
3. Use `train/train_td_graph.py` or the curriculum script when you want to study generalization to randomized mazes.

## License / Course Context

This repository appears to be coursework for an RL final project focused on robot navigation with TD-based methods. If you distribute or reuse it outside the course setting, check your team and course policy first.

## Team Members and Contributions

This project was completed by a 3-person team for the AIAA3053 Reinforcement Learning final project.

| Member | Main Contributions |
| --- | --- |
| Ye Guo | Designed and implemented the core TD(lambda) navigation methods, eligibility-trace update logic, action-conditional feature representation, and custom maze environment design. |
| Hongliang Chen | Implemented and organized baseline methods, including Q-learning and PPO-related comparison experiments, and contributed to training-script organization. |
| Junliang Huang | Conducted evaluation experiments, organized saved results, generated plots/tables, and contributed to result analysis and report revision. |

All members contributed to project discussion, debugging, presentation preparation, and final report polishing.