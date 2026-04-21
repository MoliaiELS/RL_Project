from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from utils.action_mask import get_valid_actions

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@dataclass
class MazeGraphData:
    node_features: torch.Tensor
    normalized_adjacency: torch.Tensor
    node_positions: torch.Tensor
    agent_index: int
    goal_index: int
    index_map: torch.Tensor
    action_neighbor_indices: list[int]
    valid_action_mask: np.ndarray

    @property
    def agent_position(self) -> tuple[int, int]:
        agent_pos = self.node_positions[self.agent_index]
        return int(agent_pos[0].item()), int(agent_pos[1].item())


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    adjacency_with_self = adjacency + identity
    degrees = adjacency_with_self.sum(dim=1).clamp(min=1.0)
    degree_inv_sqrt = degrees.rsqrt()
    return degree_inv_sqrt.unsqueeze(1) * adjacency_with_self * degree_inv_sqrt.unsqueeze(0)


def get_action_neighbor_indices(
    agent_pos: tuple[int, int],
    index_map: np.ndarray,
) -> list[int]:
    height, width = index_map.shape
    neighbor_indices: list[int] = []
    for dy, dx in DIRECTIONS:
        ny = agent_pos[0] + dy
        nx = agent_pos[1] + dx
        if 0 <= ny < height and 0 <= nx < width and index_map[ny, nx] >= 0:
            neighbor_indices.append(int(index_map[ny, nx]))
        else:
            neighbor_indices.append(-1)
    return neighbor_indices


def build_maze_graph(observation: np.ndarray, device: torch.device) -> MazeGraphData:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 3 or obs.shape[2] < 3:
        raise ValueError("Expected a 3-channel maze observation with wall, goal, and agent planes.")

    wall_plane = obs[:, :, 0] > 0.5
    goal_plane = obs[:, :, 1] > 0.5
    agent_plane = obs[:, :, 2] > 0.5
    height, width = wall_plane.shape

    free_positions = np.argwhere(~wall_plane)
    if free_positions.size == 0:
        raise ValueError("Maze observation contains no traversable cells.")

    index_map_np = -np.ones((height, width), dtype=np.int64)
    for node_index, (y, x) in enumerate(free_positions):
        index_map_np[y, x] = node_index

    goal_positions = np.argwhere(goal_plane)
    goal_pos = tuple(goal_positions[0].tolist()) if goal_positions.size else (-1, -1)

    agent_positions = np.argwhere(agent_plane)
    if agent_positions.size == 0:
        raise ValueError("Agent position not found in observation.")
    agent_pos = tuple(agent_positions[0].tolist())
    agent_index = int(index_map_np[agent_pos])
    if agent_index < 0:
        raise ValueError("Agent is standing on a wall cell.")

    goal_index = int(index_map_np[goal_pos]) if goal_pos[0] >= 0 else -1

    input_dim = 14
    node_features = np.zeros((len(free_positions), input_dim), dtype=np.float32)
    adjacency = np.zeros((len(free_positions), len(free_positions)), dtype=np.float32)

    for node_index, (y, x) in enumerate(free_positions):
        node_features[node_index, 0] = float(agent_plane[y, x])
        node_features[node_index, 1] = float(goal_plane[y, x])
        node_features[node_index, 2] = float(y) / max(1, height - 1)
        node_features[node_index, 3] = float(x) / max(1, width - 1)

        if goal_pos[0] >= 0:
            rel_y = float(goal_pos[0] - y) / max(1, height - 1)
            rel_x = float(goal_pos[1] - x) / max(1, width - 1)
            manhattan = (abs(goal_pos[0] - y) + abs(goal_pos[1] - x)) / max(1, height + width - 2)
            node_features[node_index, 4] = rel_y
            node_features[node_index, 5] = rel_x
            node_features[node_index, 11] = float(manhattan)
            node_features[node_index, 12] = float(goal_pos[0] == y)
            node_features[node_index, 13] = float(goal_pos[1] == x)

        blocked = 0
        for direction_index, (dy, dx) in enumerate(DIRECTIONS):
            ny = y + dy
            nx = x + dx
            is_blocked = ny < 0 or ny >= height or nx < 0 or nx >= width or wall_plane[ny, nx]
            node_features[node_index, 6 + direction_index] = float(is_blocked)
            blocked += int(is_blocked)
            if not is_blocked:
                adjacency[node_index, index_map_np[ny, nx]] = 1.0

        node_features[node_index, 10] = blocked / 4.0

    valid_action_mask = get_valid_actions(agent_pos, index_map_np)
    action_neighbor_indices = get_action_neighbor_indices(agent_pos, index_map_np)

    adjacency_tensor = torch.from_numpy(adjacency).to(device=device, dtype=torch.float32)

    return MazeGraphData(
        node_features=torch.from_numpy(node_features).to(device=device, dtype=torch.float32),
        normalized_adjacency=normalize_adjacency(adjacency_tensor),
        node_positions=torch.from_numpy(free_positions.astype(np.int64)).to(device=device),
        agent_index=agent_index,
        goal_index=goal_index,
        index_map=torch.from_numpy(index_map_np).to(device=device),
        action_neighbor_indices=action_neighbor_indices,
        valid_action_mask=valid_action_mask,
    )


class ResidualGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_messages = normalized_adjacency @ h
        h_new = self.update(torch.cat([h, neighbor_messages], dim=-1))
        return self.norm(h + h_new)


class GraphEncoder(nn.Module):
    """Goal-conditioned residual graph encoder for maze observations."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = 14
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(self.input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [ResidualGraphLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, node_features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_proj(node_features))
        h = self.input_dropout(self.input_norm(h))
        for layer in self.layers:
            h = layer(h, normalized_adjacency)
        return h


__all__ = [
    "DIRECTIONS",
    "GraphEncoder",
    "MazeGraphData",
    "build_maze_graph",
    "get_action_neighbor_indices",
    "normalize_adjacency",
]
