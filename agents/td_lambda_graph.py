import numpy as np
import torch
import torch.nn as nn
from .base_agent import BaseAgent

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GraphEncoder(nn.Module):
    """A simple cell-level graph encoder for maze observations."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = 11
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(self.input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(node_features)
        for layer in self.layers:
            neighbor_messages = adjacency @ h
            h = torch.cat([h, neighbor_messages], dim=-1)
            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def build_graph(self, observation: np.ndarray, device: torch.device):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise ValueError(
                "Expected a 3-channel maze observation with wall, goal, and agent planes."
            )

        wall_plane = obs[:, :, 0] > 0.5
        goal_plane = obs[:, :, 1] > 0.5
        agent_plane = obs[:, :, 2] > 0.5
        height, width = wall_plane.shape

        free_positions = np.argwhere(~wall_plane)
        if free_positions.size == 0:
            raise ValueError("Maze observation contains no traversable cells.")

        node_count = len(free_positions)
        index_map = -np.ones((height, width), dtype=np.int64)
        for node_index, (y, x) in enumerate(free_positions):
            index_map[y, x] = node_index

        node_features = np.zeros((node_count, self.input_dim), dtype=np.float32)
        goal_positions = np.argwhere(goal_plane)
        goal_pos = tuple(goal_positions[0].tolist()) if goal_positions.size else (-1, -1)

        for node_index, (y, x) in enumerate(free_positions):
            node_features[node_index, 0] = float(agent_plane[y, x])
            node_features[node_index, 1] = float(goal_plane[y, x])
            node_features[node_index, 2] = float(y) / max(1, height - 1)
            node_features[node_index, 3] = float(x) / max(1, width - 1)
            if goal_pos[0] >= 0:
                node_features[node_index, 4] = float(goal_pos[0] - y) / max(1, height - 1)
                node_features[node_index, 5] = float(goal_pos[1] - x) / max(1, width - 1)
            else:
                node_features[node_index, 4] = 0.0
                node_features[node_index, 5] = 0.0

            blocked = 0
            for direction_index, (dy, dx) in enumerate(DIRECTIONS):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width or wall_plane[ny, nx]:
                    node_features[node_index, 6 + direction_index] = 1.0
                    blocked += 1
                else:
                    node_features[node_index, 6 + direction_index] = 0.0

            node_features[node_index, 10] = float(blocked) / 4.0

        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        for node_index, (y, x) in enumerate(free_positions):
            adjacency[node_index, node_index] = 1.0
            for dy, dx in DIRECTIONS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and not wall_plane[ny, nx]:
                    neighbor_index = index_map[ny, nx]
                    if neighbor_index >= 0:
                        adjacency[node_index, neighbor_index] = 1.0

        agent_positions = np.argwhere(agent_plane)
        if agent_positions.size == 0:
            raise ValueError("Agent position not found in observation.")
        agent_coord = tuple(agent_positions[0].tolist())
        agent_index = int(index_map[agent_coord])
        if agent_index < 0:
            raise ValueError("Agent is standing on a wall cell.")

        goal_index = int(index_map[goal_pos]) if goal_pos[0] >= 0 else -1

        return (
            torch.from_numpy(node_features).to(device),
            torch.from_numpy(adjacency).to(device),
            torch.from_numpy(free_positions.astype(np.int64)).to(device),
            agent_index,
            goal_index,
            torch.from_numpy(index_map).to(device),
        )


class ActionValueHead(nn.Module):
    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_actions + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        agent_embedding: torch.Tensor,
        neighbor_embedding: torch.Tensor,
        action_onehot: torch.Tensor,
        invalid_flag: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([agent_embedding, neighbor_embedding, action_onehot, invalid_flag], dim=-1)
        return self.net(x).squeeze(-1)


class TDLambdaGraphAgent(BaseAgent):
    """Graph-based TD(λ) agent for maze navigation."""

    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 5e-4,
        epsilon: float = 0.1,
        lambda_value: float = 0.9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        seed: int | None = None,
        device: str | None = None,
    ):
        state_size = hidden_dim
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.lambda_value = float(lambda_value)
        self.hidden_dim = hidden_dim
        self.device = (
            torch.device("cuda")
            if device is None and torch.cuda.is_available()
            else torch.device(device or "cpu")
        )
        self.encoder = GraphEncoder(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = ActionValueHead(hidden_dim=hidden_dim, n_actions=n_actions)
        self.model = nn.ModuleDict({"encoder": self.encoder, "head": self.head})
        self.model.to(self.device)
        self.eligibility = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]

    def _action_onehot(self, action: int) -> torch.Tensor:
        onehot = torch.zeros(self.n_actions, device=self.device)
        if 0 <= action < self.n_actions:
            onehot[action] = 1.0
        return onehot

    def _compute_q_values(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_positions: torch.Tensor,
        agent_index: int,
        index_map: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.encoder(node_features, adjacency)
        agent_embedding = node_embeddings[agent_index : agent_index + 1]
        agent_pos = tuple(node_positions[agent_index].tolist())

        q_values = []
        for action in range(self.n_actions):
            ny = int(agent_pos[0] + DIRECTIONS[action][0])
            nx = int(agent_pos[1] + DIRECTIONS[action][1])
            invalid = True
            neighbor_embedding = torch.zeros((1, self.hidden_dim), device=self.device)
            if 0 <= ny < index_map.shape[0] and 0 <= nx < index_map.shape[1]:
                neighbor_index = int(index_map[ny, nx].item())
                if neighbor_index >= 0:
                    invalid = False
                    neighbor_embedding = node_embeddings[neighbor_index : neighbor_index + 1]

            invalid_flag = torch.tensor([[1.0 if invalid else 0.0]], device=self.device)
            action_onehot = self._action_onehot(action).unsqueeze(0)
            q_value = self.head(agent_embedding, neighbor_embedding, action_onehot, invalid_flag)
            if invalid:
                q_value = q_value - 5.0
            q_values.append(q_value)

        return torch.stack(q_values).squeeze(-1)

    def _build_graph(self, observation: np.ndarray):
        return self.encoder.build_graph(observation, self.device)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            node_features, adjacency, node_positions, agent_index, _, index_map = self._build_graph(state)
            q_values = self._compute_q_values(node_features, adjacency, node_positions, agent_index, index_map)
            return q_values.cpu().numpy()

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.argmax_action(self.q_values(state))

    def new_episode(self) -> None:
        self.eligibility = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        node_features, adjacency, node_positions, agent_index, _, index_map = self._build_graph(state)
        current_q_values = self._compute_q_values(node_features, adjacency, node_positions, agent_index, index_map)
        current_q = current_q_values[action]

        if done or next_action is None:
            target = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        else:
            with torch.no_grad():
                next_node_features, next_adjacency, next_node_positions, next_agent_index, _, next_index_map = self._build_graph(next_state)
                next_q_values = self._compute_q_values(
                    next_node_features,
                    next_adjacency,
                    next_node_positions,
                    next_agent_index,
                    next_index_map,
                )
                target = float(reward) + self.gamma * next_q_values[next_action]

        td_error = target - current_q
        self.model.zero_grad()
        current_q.backward()

        with torch.no_grad():
            for idx, param in enumerate(self.model.parameters()):
                if param.grad is None:
                    continue
                grad = param.grad.detach().clamp(-5.0, 5.0)
                self.eligibility[idx] = self.gamma * self.lambda_value * self.eligibility[idx] + grad
                self.eligibility[idx].clamp_(-10.0, 10.0)
                param.add_(self.alpha * td_error * self.eligibility[idx])

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


__all__ = ["GraphEncoder", "TDLambdaGraphAgent"]
