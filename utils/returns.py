from __future__ import annotations

from typing import Callable

import numpy as np


def compute_n_step_return(
    transitions: list[dict],
    start_index: int,
    n: int,
    gamma: float,
    bootstrap_q: Callable[[np.ndarray, int], float] | None = None,
) -> float:
    """Compute the truncated n-step return starting at start_index."""
    return_value = 0.0
    gamma_power = 1.0
    last_index = start_index + n - 1
    for idx in range(start_index, start_index + n):
        if idx >= len(transitions):
            break
        reward = transitions[idx]["reward"]
        return_value += gamma_power * reward
        gamma_power *= gamma
        if transitions[idx]["done"]:
            return return_value
    if bootstrap_q is not None and last_index < len(transitions) and not transitions[last_index]["done"]:
        next_action = transitions[last_index].get("next_action")
        if next_action is not None:
            next_state = transitions[last_index]["next_state"]
            return_value += gamma_power * bootstrap_q(next_state, next_action)
    return return_value


def compute_lambda_returns(
    transitions: list[dict],
    gamma: float,
    lambda_value: float,
    bootstrap_q: Callable[[np.ndarray, int], float] | None = None,
) -> list[float]:
    """Compute forward-view lambda-return targets for a trajectory."""
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"lambda_value must be in [0, 1], got {lambda_value}")

    n_steps = len(transitions)
    lambda_returns = []
    for t in range(n_steps):
        n_step_returns = []
        for n in range(1, n_steps - t + 1):
            n_step = compute_n_step_return(transitions, t, n, gamma, bootstrap_q)
            n_step_returns.append(n_step)
        # The last term is included with weight lambda^{N-1}
        weights = [(1 - lambda_value) * (lambda_value ** i) for i in range(len(n_step_returns))]
        if weights:
            weights[-1] = weights[-1] + lambda_value ** (len(n_step_returns) - 1)
        lambda_return = sum(w * g for w, g in zip(weights, n_step_returns))
        lambda_returns.append(lambda_return)
    return lambda_returns
