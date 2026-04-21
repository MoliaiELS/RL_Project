from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
import torch.nn as nn

from utils.action_mask import masked_q_values, select_action_from_valid_actions
from utils.returns import compute_lambda_returns

from .base_agent import BaseAgent
from .graph_encoder import GraphEncoder, MazeGraphData, build_maze_graph


class ActionValueHead(nn.Module):
    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_actions + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        agent_embedding: torch.Tensor,
        neighbor_embedding: torch.Tensor,
        action_onehot: torch.Tensor,
        invalid_flag: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [agent_embedding, neighbor_embedding, action_onehot, invalid_flag],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class GraphActionValueNetwork(nn.Module):
    def __init__(
        self,
        n_actions: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.encoder = GraphEncoder(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = ActionValueHead(hidden_dim=hidden_dim, n_actions=n_actions)
        self.register_buffer("action_eye", torch.eye(n_actions, dtype=torch.float32))

    def forward(self, graph: MazeGraphData) -> torch.Tensor:
        node_embeddings = self.encoder(graph.node_features, graph.normalized_adjacency)
        agent_embedding = node_embeddings[graph.agent_index : graph.agent_index + 1]
        zero_neighbor = node_embeddings.new_zeros((1, self.hidden_dim))

        q_values = []
        for action in range(self.n_actions):
            neighbor_index = graph.action_neighbor_indices[action]
            neighbor_embedding = (
                node_embeddings[neighbor_index : neighbor_index + 1]
                if neighbor_index >= 0
                else zero_neighbor
            )
            invalid_flag = node_embeddings.new_tensor([[1.0 if neighbor_index < 0 else 0.0]])
            action_onehot = self.action_eye[action : action + 1]
            q_values.append(
                self.head(
                    agent_embedding,
                    neighbor_embedding,
                    action_onehot,
                    invalid_flag,
                )
            )

        return torch.cat(q_values, dim=0)


class TDLambdaGraphAgent(BaseAgent):
    """Graph-based forward-view TD(lambda) agent with masked action selection."""

    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 2e-4,
        epsilon: float = 0.3,
        lambda_value: float = 0.7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        seed: int | None = None,
        device: str | None = None,
        use_target_network: bool = True,
        target_tau: float = 0.005,
        target_update_freq: int = 1,
        batch_size: int = 64,
        gradient_clip: float = 1.0,
    ):
        super().__init__(hidden_dim, n_actions, gamma, alpha, epsilon, seed)
        self.lambda_value = float(lambda_value)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.device = self._resolve_device(device)
        self.use_target_network = bool(use_target_network)
        self.target_tau = float(target_tau)
        self.target_update_freq = max(1, int(target_update_freq))
        self.batch_size = max(1, int(batch_size))
        self.gradient_clip = float(gradient_clip)
        self.update_mode = "lambda_return"

        self.online_network = GraphActionValueNetwork(
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.target_network = copy.deepcopy(self.online_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.alpha)
        self.optimizer_steps = 0

        # Compatibility handles for older scripts/checkpoints.
        self.model = self.online_network
        self.encoder = self.online_network.encoder
        self.head = self.online_network.head

    def _resolve_device(self, device: str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            resolved = torch.device(device)
        except Exception as exc:
            warnings.warn(f"Invalid device '{device}', falling back to CPU. ({exc})")
            return torch.device("cpu")

        if resolved.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA was requested but is unavailable, falling back to CPU.")
            return torch.device("cpu")
        return resolved

    def _build_graph(self, observation: np.ndarray) -> MazeGraphData:
        return build_maze_graph(observation, self.device)

    def _q_values_tensor(
        self,
        state: np.ndarray,
        network: GraphActionValueNetwork | None = None,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor, MazeGraphData]:
        graph = self._build_graph(state)
        model = network or self.online_network
        if require_grad:
            q_values = model(graph)
        else:
            with torch.no_grad():
                q_values = model(graph)
        return q_values, graph

    def q_values(self, state: np.ndarray) -> np.ndarray:
        q_values, _ = self._q_values_tensor(state, require_grad=False)
        return q_values.detach().cpu().numpy()

    def masked_q_values(self, state: np.ndarray) -> np.ndarray:
        q_values, graph = self._q_values_tensor(state, require_grad=False)
        return masked_q_values(q_values.detach().cpu().numpy(), graph.valid_action_mask)

    def select_action_details(
        self,
        state: np.ndarray,
        greedy: bool = False,
    ) -> tuple[int, dict]:
        q_values, graph = self._q_values_tensor(state, require_grad=False)
        q_values_np = q_values.detach().cpu().numpy()
        if greedy:
            action = select_action_from_valid_actions(
                q_values_np,
                graph.valid_action_mask,
                self.rng,
                greedy=True,
            )
        else:
            explore = self.rng.random() < self.epsilon
            action = select_action_from_valid_actions(
                q_values_np,
                graph.valid_action_mask,
                self.rng,
                greedy=not explore,
            )

        return action, {
            "q_values": q_values_np,
            "masked_q_values": masked_q_values(q_values_np, graph.valid_action_mask),
            "valid_action_mask": graph.valid_action_mask.copy(),
            "valid_action_count": int(np.count_nonzero(graph.valid_action_mask)),
            "selected_valid": bool(graph.valid_action_mask[action]) if np.any(graph.valid_action_mask) else False,
            "used_fallback": bool(not np.any(graph.valid_action_mask)),
        }

    def select_action(self, state: np.ndarray) -> int:
        action, _ = self.select_action_details(state, greedy=False)
        return action

    def greedy_action(self, state: np.ndarray) -> int:
        action, _ = self.select_action_details(state, greedy=True)
        return action

    def new_episode(self) -> None:
        return None

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> dict:
        return {
            "state": np.array(state, copy=True),
            "action": int(action),
            "reward": float(reward),
            "next_state": None if next_state is None else np.array(next_state, copy=True),
            "done": bool(done),
            "next_action": None if next_action is None else int(next_action),
        }

    def _bootstrap_q_value(self, state: np.ndarray, action: int) -> float:
        model = self.target_network if self.use_target_network else self.online_network
        q_values, _ = self._q_values_tensor(state, network=model, require_grad=False)
        return float(q_values[action].item())

    def compute_lambda_returns(self, transitions: list[dict]) -> list[float]:
        return compute_lambda_returns(
            transitions,
            gamma=self.gamma,
            lambda_value=self.lambda_value,
            bootstrap_q=self._bootstrap_q_value,
        )

    def _predict_action_value(self, state: np.ndarray, action: int) -> torch.Tensor:
        q_values, _ = self._q_values_tensor(state, require_grad=True)
        return q_values[action]

    def _soft_update_target_network(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.online_network.parameters(),
            ):
                target_param.data.mul_(1.0 - self.target_tau).add_(self.target_tau * online_param.data)

    def update_from_episode(self, transitions: list[dict]) -> dict:
        if not transitions:
            return {
                "loss": 0.0,
                "mean_predicted_q": 0.0,
                "mean_td_target": 0.0,
                "optimizer_steps": 0,
            }

        lambda_returns = self.compute_lambda_returns(transitions)
        indices = np.arange(len(transitions))
        batch_size = min(self.batch_size, len(transitions))

        losses: list[float] = []
        predicted_means: list[float] = []
        target_means: list[float] = []
        optimizer_steps = 0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            targets = torch.tensor(
                [lambda_returns[idx] for idx in batch_indices],
                dtype=torch.float32,
                device=self.device,
            )
            predictions = torch.stack(
                [
                    self._predict_action_value(
                        transitions[idx]["state"],
                        transitions[idx]["action"],
                    )
                    for idx in batch_indices
                ]
            )

            loss = nn.functional.mse_loss(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.gradient_clip)
            self.optimizer.step()

            optimizer_steps += 1
            self.optimizer_steps += 1
            losses.append(float(loss.item()))
            predicted_means.append(float(predictions.detach().mean().item()))
            target_means.append(float(targets.mean().item()))

            if self.use_target_network and self.optimizer_steps % self.target_update_freq == 0:
                self._soft_update_target_network()

        return {
            "loss": float(np.mean(losses)),
            "mean_predicted_q": float(np.mean(predicted_means)),
            "mean_td_target": float(np.mean(target_means)),
            "optimizer_steps": optimizer_steps,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online_network.state_dict(),
                "target": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "n_actions": self.n_actions,
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                    "epsilon": self.epsilon,
                    "lambda_value": self.lambda_value,
                    "use_target_network": self.use_target_network,
                    "target_tau": self.target_tau,
                    "target_update_freq": self.target_update_freq,
                    "batch_size": self.batch_size,
                    "gradient_clip": self.gradient_clip,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "online" in checkpoint:
            self.online_network.load_state_dict(checkpoint["online"])
            self.target_network.load_state_dict(checkpoint.get("target", checkpoint["online"]))
            if "optimizer" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except ValueError:
                    warnings.warn("Skipping optimizer state load because it is incompatible with the current agent.")
        else:
            self.online_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(checkpoint)

        self.model = self.online_network
        self.encoder = self.online_network.encoder
        self.head = self.online_network.head


__all__ = ["ActionValueHead", "GraphActionValueNetwork", "TDLambdaGraphAgent"]
