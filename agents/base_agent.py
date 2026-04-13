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

    def greedy_action(self, state: np.ndarray) -> int:
        q_values = self.q_values(state)
        return int(np.argmax(q_values))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


__all__ = ["BaseAgent"]
