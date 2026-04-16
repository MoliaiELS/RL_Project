import numpy as np
import torch
import torch.nn as nn
from .base_agent import BaseAgent


class CNNQNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        channels = obs_shape[2]
        height, width = obs_shape[0], obs_shape[1]
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * height * width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TDLambdaCNNAgent(BaseAgent):
    """TD(λ) agent using a CNN on raw maze observations."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 5e-4,
        epsilon: float = 0.1,
        lambda_value: float = 0.9,
        seed: int | None = None,
    ):
        self.obs_shape = obs_shape
        state_size = int(np.prod(obs_shape))
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.lambda_value = float(lambda_value)
        self.device = torch.device("cpu")
        self.net = CNNQNetwork(obs_shape, n_actions).to(self.device)
        self.eligibility = [torch.zeros_like(param.data) for param in self.net.parameters()]

    def _to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        x = np.asarray(observation, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != self.obs_shape[2]:
            raise ValueError(
                f"Expected raw observation shape {self.obs_shape}, got {x.shape}."
            )
        x = torch.from_numpy(x).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self._to_tensor(state)
            q = self.net(x)[0]
        return q.cpu().numpy()

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.argmax_action(self.q_values(state))

    def new_episode(self) -> None:
        self.eligibility = [torch.zeros_like(param.data) for param in self.net.parameters()]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        state_tensor = self._to_tensor(state)
        self.net.zero_grad()
        q_values = self.net(state_tensor)[0]
        current_q = q_values[action]

        if done or next_action is None:
            target = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        else:
            with torch.no_grad():
                next_q_values = self.net(self._to_tensor(next_state))[0]
                target = float(reward) + self.gamma * next_q_values[next_action]

        td_error = target - current_q
        current_q.backward()

        with torch.no_grad():
            for idx, param in enumerate(self.net.parameters()):
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                self.eligibility[idx] = self.gamma * self.lambda_value * self.eligibility[idx] + grad
                param.add_(self.alpha * td_error * self.eligibility[idx])

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))


__all__ = ["TDLambdaCNNAgent"]
