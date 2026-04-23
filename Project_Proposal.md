# Project Proposal

## Title
**Representation-Aware TD(lambda) for Robot Navigation in Procedurally Generated Mazes**

## 1. Background and Motivation
Robot navigation is a natural reinforcement learning problem because an agent must learn a sequence of decisions that balances short-term movement costs with long-term goal reaching. In grid-based mazes, the challenge is not only to find a path, but also to generalize across layouts with dead ends, narrow corridors, and dynamically changing obstacle structures. This makes maze navigation a good testbed for studying whether improved state representations can make temporal-difference learning more effective.

Our current codebase already implements a complete navigation benchmark around this idea. It includes custom maze environments with reward shaping, support for fixed and procedurally generated mazes, linear TD(0), linear TD(lambda), a TD(lambda) agent with handcrafted action-conditional navigation features, a CNN-based TD(lambda) agent that learns directly from raw maze observations, and comparison baselines using Q-learning and PPO. Because these components are already available, the project can move beyond a basic implementation exercise and focus on a meaningful question: how much does representation choice matter for TD-based control in robot navigation?

## 2. Problem Statement
This project investigates whether richer state-action representations improve the learning efficiency and generalization ability of TD(lambda) agents for maze navigation. The task is for an agent to move from a start cell to a goal cell in grid mazes while avoiding walls and minimizing unnecessary motion. The environments include both hand-designed mazes such as `Maze-Easy`, `Maze-Medium`, and `Maze-Hard`, and auto-generated environments such as `Maze-Auto` and `Maze-Auto-Random-9x9`.

The core problem is that simple flattened observations may be too weak to capture the local spatial structure needed for efficient navigation. A linear TD(lambda) agent can learn from these observations, but it may struggle in procedurally generated mazes where the geometry changes frequently. We therefore propose to compare three TD-based representations:

1. A flat linear representation over encoded observations.
2. A handcrafted action-conditional feature representation that explicitly models invalid moves, distance-to-goal changes, local wall patches, and dead-end heuristics.
3. A CNN-based representation that learns action values directly from raw maze observations.

## 3. Objectives and Research Questions
The main objective is to build and evaluate a family of TD-based navigation agents and determine which representation is most effective under different maze settings.

The project is guided by the following research questions:

- Does TD(lambda) outperform TD(0) and linear Q-learning in maze navigation when eligibility traces are used to propagate delayed rewards?
- Do action-conditional handcrafted features improve sample efficiency compared with flat observation encoding?
- Can a CNN-based TD(lambda) agent better exploit raw spatial structure in procedurally generated mazes?
- How do TD-based agents compare with PPO as a stronger policy-gradient baseline on the same navigation tasks?
- Does curriculum-style training from fixed mazes to randomized mazes improve robustness and generalization?

## 4. Methodology
The environment is a custom Gymnasium-compatible maze domain. Each maze observation is represented as a 3-channel tensor containing wall, goal, and agent locations. The reward is shaped to support learning: reaching the goal gives a positive terminal reward, invalid moves are penalized, and each step receives either a small time penalty or a distance-based shaping term derived from Manhattan distance to the goal.

The experimental pipeline is already implemented in the repository:

- [`train/train_td.py`](c:\Users\moliai\Desktop\RL_Project\train\train_td.py) trains `td0` and linear `tdlambda` agents on both Maze and MiniGrid environments.
- [`train/train_td_action_features.py`](c:\Users\moliai\Desktop\RL_Project\train\train_td_action_features.py) trains a TD(lambda) agent with explicit action-conditional navigation features.
- [`train/train_td_cnn.py`](c:\Users\moliai\Desktop\RL_Project\train\train_td_cnn.py) trains a CNN-based TD(lambda) agent on raw maze observations.
- [`train/train_q.py`](c:\Users\moliai\Desktop\RL_Project\train\train_q.py) and [`train/train_ppo.py`](c:\Users\moliai\Desktop\RL_Project\train\train_ppo.py) provide baseline comparisons.
- [`eval/evaluate.py`](c:\Users\moliai\Desktop\RL_Project\eval\evaluate.py) supports greedy evaluation, replay generation, and summary statistics for the current TD-family agents.

We will run experiments in three stages. First, we will establish baseline learning curves on fixed mazes such as `Maze-Easy` and `Maze-Hard`. Second, we will evaluate transfer to generated mazes such as `Maze-Auto-9x9` and `Maze-Auto-Random-9x9`. Third, we will study curriculum-style training using the existing pretraining and parameter search scripts to move from stable fixed layouts to randomized layouts.

The primary metrics will be average episode reward, greedy evaluation reward, success consistency across seeds, and qualitative replay behavior. We will also compare convergence speed by examining how quickly each method reaches stable positive or near-goal performance.

## 5. Preliminary Evidence and Feasibility
This project is highly feasible because most of the system has already been implemented and tested. The repository contains saved training runs, metadata logging, reward curves, and evaluation histories. In one existing action-feature curriculum run on `Maze-Auto-Random-9x9`, the best greedy evaluation mean reached about **0.588** at episode **25,750**, although the final evaluation remained unstable at about **-2.75** by episode **30,000**. This is an encouraging result because it shows the agent can sometimes solve randomized mazes, but it also reveals a clear research challenge: performance is not yet consistently robust.

That instability is exactly what makes the project interesting. Rather than merely showing that TD(lambda) can be coded, the project can analyze when and why different representations succeed or fail. The presence of both stronger representations and stronger baselines means the final report can include meaningful comparisons instead of only a single-agent demonstration.

## 6. Expected Contributions
We expect the project to make three contributions. First, it will provide a clean comparison of TD-based control methods for robot navigation under a common experimental framework. Second, it will show how representation design affects temporal-difference learning, especially under randomized maze generation. Third, it will produce a reproducible benchmark with interpretable baselines, saved models, evaluation plots, and replay tools that can support further extensions after the course project.

## 7. Timeline
In the short term, we will finalize the baseline experiments for TD(0), TD(lambda), Q-learning, and PPO on fixed and random mazes. Next, we will tune the action-feature and CNN TD(lambda) agents using the existing scripts for parameter search and curriculum training. Finally, we will consolidate the results into comparative plots, analyze agent behavior using replay files, and write the final report around performance, representation quality, and generalization.
