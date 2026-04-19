import numpy as np
import torch
import torch.nn as nn
from torch import sparse
from .base_agent import BaseAgent

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GraphEncoderGPU(nn.Module):
    """GPU-accelerated graph encoder for maze observations."""

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
            # Handle both sparse and dense adjacency matrices
            if adjacency.is_sparse:
                neighbor_messages = torch.sparse.mm(adjacency, h)
            else:
                neighbor_messages = adjacency @ h
            h = torch.cat([h, neighbor_messages], dim=-1)
            h = layer(h)
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def build_graph_gpu(self, observation: np.ndarray, device: torch.device):
        """GPU-accelerated graph construction using PyTorch operations."""
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device)
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise ValueError(
                "Expected a 3-channel maze observation with wall, goal, and agent planes."
            )

        wall_plane = obs[:, :, 0] > 0.5
        goal_plane = obs[:, :, 1] > 0.5
        agent_plane = obs[:, :, 2] > 0.5
        height, width = wall_plane.shape

        # Find free positions using PyTorch
        free_mask = ~wall_plane
        free_positions = torch.nonzero(free_mask, as_tuple=False)
        if free_positions.size(0) == 0:
            raise ValueError("Maze observation contains no traversable cells.")

        node_count = free_positions.size(0)

        # Create index map using PyTorch
        index_map = torch.full((height, width), -1, dtype=torch.long, device=device)
        node_indices = torch.arange(node_count, device=device)
        index_map[free_positions[:, 0], free_positions[:, 1]] = node_indices

        # Build node features using vectorized operations
        node_features = torch.zeros((node_count, self.input_dim), dtype=torch.float32, device=device)

        # Find goal position
        goal_positions = torch.nonzero(goal_plane, as_tuple=False)
        goal_pos = goal_positions[0] if goal_positions.size(0) > 0 else torch.tensor([-1, -1], device=device)

        # Agent and goal flags
        node_features[:, 0] = agent_plane[free_positions[:, 0], free_positions[:, 1]].float()
        node_features[:, 1] = goal_plane[free_positions[:, 0], free_positions[:, 1]].float()

        # Normalized positions
        node_features[:, 2] = free_positions[:, 0].float() / max(1, height - 1)
        node_features[:, 3] = free_positions[:, 1].float() / max(1, width - 1)

        # Distance to goal (if goal exists)
        if goal_pos[0] >= 0:
            node_features[:, 4] = (goal_pos[0] - free_positions[:, 0]).float() / max(1, height - 1)
            node_features[:, 5] = (goal_pos[1] - free_positions[:, 1]).float() / max(1, width - 1)

        # Check walls in each direction using vectorized operations
        for direction_index, (dy, dx) in enumerate(DIRECTIONS):
            ny = free_positions[:, 0] + dy
            nx = free_positions[:, 1] + dx

            # Check bounds and walls
            in_bounds = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            wall_blocked = torch.zeros_like(in_bounds, dtype=torch.bool, device=device)
            wall_blocked[in_bounds] = wall_plane[ny[in_bounds], nx[in_bounds]]

            blocked = ~in_bounds | wall_blocked
            node_features[:, 6 + direction_index] = blocked.float()

        # Wall density
        node_features[:, 10] = node_features[:, 6:10].sum(dim=1) / 4.0

        # Build sparse adjacency matrix
        adj_indices = []
        adj_values = []

        # Self-connections
        adj_indices.append(torch.stack([node_indices, node_indices]))
        adj_values.append(torch.ones(node_count, device=device))

        # Neighbor connections
        for dy, dx in DIRECTIONS:
            ny = free_positions[:, 0] + dy
            nx = free_positions[:, 1] + dx

            in_bounds = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            not_wall = torch.zeros_like(in_bounds, dtype=torch.bool, device=device)
            not_wall[in_bounds] = ~wall_plane[ny[in_bounds], nx[in_bounds]]

            valid_neighbors = in_bounds & not_wall
            if valid_neighbors.any():
                neighbor_indices = index_map[ny[valid_neighbors], nx[valid_neighbors]]
                source_indices = node_indices[valid_neighbors]

                adj_indices.append(torch.stack([source_indices, neighbor_indices]))
                adj_values.append(torch.ones(source_indices.size(0), device=device))

        # Combine all adjacency entries
        if adj_indices:
            all_indices = torch.cat(adj_indices, dim=1)
            all_values = torch.cat(adj_values)
            # Try to create sparse tensor, fallback to dense if not available
            try:
                adjacency = torch.sparse_coo_tensor(
                    all_indices,
                    all_values,
                    (node_count, node_count),
                    dtype=torch.float32,
                    device=device
                )
            except (AttributeError, TypeError):
                # Fallback for older PyTorch versions
                adjacency = torch.zeros((node_count, node_count), dtype=torch.float32, device=device)
                adjacency[all_indices[0], all_indices[1]] = all_values
        else:
            try:
                adjacency = torch.sparse_coo_tensor(
                    torch.empty(2, 0, dtype=torch.long, device=device),
                    torch.empty(0, device=device),
                    (node_count, node_count),
                    dtype=torch.float32,
                    device=device
                )
            except (AttributeError, TypeError):
                # Fallback for older PyTorch versions
                adjacency = torch.zeros((node_count, node_count), dtype=torch.float32, device=device)

        # Find agent and goal indices
        agent_positions = torch.nonzero(agent_plane, as_tuple=False)
        if agent_positions.size(0) == 0:
            raise ValueError("Agent position not found in observation.")
        agent_coord = agent_positions[0]
        agent_index = index_map[agent_coord[0], agent_coord[1]]
        if agent_index < 0:
            raise ValueError("Agent is standing on a wall cell.")

        goal_index = index_map[goal_pos[0], goal_pos[1]] if goal_pos[0] >= 0 else -1

        return (
            node_features,
            adjacency,
            free_positions,
            agent_index.item(),
            goal_index.item() if goal_index >= 0 else -1,
            index_map,
        )


class ActionValueHeadGPU(nn.Module):
    """GPU-accelerated action value head."""
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


class TDLambdaGraphAgentGPU(BaseAgent):
    """GPU-accelerated graph-based TD(λ) agent for maze navigation."""

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
        self.encoder = GraphEncoderGPU(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = ActionValueHeadGPU(hidden_dim=hidden_dim, n_actions=n_actions)
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
        agent_pos = node_positions[agent_index]

        q_values = []
        for action in range(self.n_actions):
            ny = int(agent_pos[0] + DIRECTIONS[action][0])
            nx = int(agent_pos[1] + DIRECTIONS[action][1])
            invalid = True
            neighbor_embedding = torch.zeros((1, self.hidden_dim), device=self.device)
            if 0 <= ny < index_map.shape[0] and 0 <= nx < index_map.shape[1]:
                neighbor_index = index_map[ny, nx].item()
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
        return self.encoder.build_graph_gpu(observation, self.device)

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


__all__ = ["GraphEncoderGPU", "TDLambdaGraphAgentGPU"]