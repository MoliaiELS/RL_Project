import numpy as np
from .base_agent import BaseAgent


class TDLambdaAgent(BaseAgent):
    """Linear TD(λ) agent with accumulating eligibility traces.

    This implementation is custom and does not rely on external RL libraries.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 5e-4,
        epsilon: float = 0.1,
        lambda_value: float = 0.9,
        seed: int | None = None,
    ):
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.lambda_value = float(lambda_value)
        self.weights = np.zeros((n_actions, state_size), dtype=np.float32)
        self.eligibility = np.zeros_like(self.weights)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights.dot(state)

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.argmax_action(self.q_values(state))

    def new_episode(self) -> None:
        self.eligibility.fill(0.0)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        current_q = self.q_values(state)[action]
        if done or next_action is None:
            target = reward
        else:
            target = reward + self.gamma * self.q_values(next_state)[next_action]
        td_error = target - current_q
        self.eligibility *= self.gamma * self.lambda_value
        self.eligibility[action] += state
        self.weights += self.alpha * td_error * self.eligibility

    def save(self, path: str) -> None:
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        self.weights = np.load(path)


__all__ = ["TDLambdaAgent"]
