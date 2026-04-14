import numpy as np
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Linear Q‑Learning agent (off‑policy TD control).

    Uses linear function approximation: Q(s,a) = w_a · φ(s).
    Implements the standard Q‑learning update:
        w_a ← w_a + α * (r + γ * max_a' Q(s',a') - Q(s,a)) * φ(s)
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 1e-3,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.weights = np.zeros((n_actions, state_size), dtype=np.float32)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q‑values for all actions given state."""
        return self.weights.dot(state)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon‑greedy action selection."""
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
        """Perform one Q‑learning update."""
        current_q = self.q_values(state)[action]

        if done:
            target = reward
        else:
            next_q_max = np.max(self.q_values(next_state))
            target = reward + self.gamma * next_q_max

        td_error = target - current_q
        self.weights[action] += self.alpha * td_error * state

    def save(self, path: str) -> None:
        """Save weights to a .npy file."""
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        """Load weights from a .npy file."""
        self.weights = np.load(path)


__all__ = ["QLearningAgent"]