import numpy as np
from .base_agent import BaseAgent


class TDZeroAgent(BaseAgent):
    def __init__(
        self,
        state_size: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 1e-5,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.weights = np.zeros((n_actions, state_size), dtype=np.float32)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights.dot(state)

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.argmax_action(self.q_values(state))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        current_q = self.q_values(state)[action]
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_values(next_state))
        td_error = target - current_q
        self.weights[action] += self.alpha * td_error * state

    def save(self, path: str) -> None:
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        self.weights = np.load(path)


__all__ = ["TDZeroAgent"]
