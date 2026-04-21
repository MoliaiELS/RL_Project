import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(
        self,
        state_size: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 1e-3,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def argmax_action(self, q_values: np.ndarray, valid_actions: np.ndarray | None = None) -> int:
        if valid_actions is not None:
            if valid_actions.size != q_values.size:
                raise ValueError("valid_actions mask and q_values must have same shape")
            masked_q = np.array(q_values, dtype=float, copy=True)
            if not np.any(valid_actions):
                masked_q = np.array(q_values, dtype=float, copy=True)
            else:
                masked_q[~valid_actions] = float("-inf")
            q_values = masked_q

        if np.all(np.isnan(q_values)):
            return int(self.rng.integers(self.n_actions))
        max_value = np.nanmax(q_values)
        max_indices = np.flatnonzero(q_values == max_value)
        if max_indices.size == 1:
            return int(max_indices[0])
        return int(self.rng.choice(max_indices))

    def greedy_action(self, state: np.ndarray, valid_actions: np.ndarray | None = None) -> int:
        q_values = self.q_values(state)
        return self.argmax_action(q_values, valid_actions)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


__all__ = ["BaseAgent"]
