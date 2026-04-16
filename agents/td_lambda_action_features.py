import numpy as np
from .base_agent import BaseAgent


class ActionConditionalFeatureExtractor:
    """Extracts explicit navigation features for a candidate action.

    The feature vector is built from the candidate next position of action a,
    including invalid-move signals, goal proximity, local wall/goal patch,
    neighbor counts, and dead-end heuristics.
    """

    def __init__(self, patch_radius: int = 1):
        self.patch_radius = patch_radius
        self.patch_size = patch_radius * 2 + 1

    @property
    def feature_size(self) -> int:
        # bias + invalid + goal + distance_delta + neighbor_ratio + dead_end
        # + wall patch + goal patch
        return 6 + 2 * (self.patch_size ** 2)

    def extract(self, observation: np.ndarray, action: int) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise ValueError(
                "Expected a 3-channel maze observation with wall, goal, and agent planes."
            )

        wall_channel = obs[:, :, 0]
        goal_channel = obs[:, :, 1]
        agent_channel = obs[:, :, 2]

        agent_pos = self._find_position(agent_channel, "agent")
        goal_pos = self._find_position(goal_channel, "goal")
        next_pos = self._next_position(agent_pos, action)

        invalid_move = float(not self._is_free(wall_channel, next_pos))
        current_distance = self._manhattan_distance(agent_pos, goal_pos)
        next_distance = (
            current_distance
            if invalid_move
            else self._manhattan_distance(next_pos, goal_pos)
        )
        distance_delta = float(current_distance - next_distance)
        next_is_goal = float(next_pos == goal_pos and not invalid_move)
        neighbor_free_count = self._count_free_neighbors(wall_channel, next_pos)
        neighbor_ratio = float(neighbor_free_count / 4.0)
        is_dead_end = float(neighbor_free_count <= 1 and next_pos != goal_pos)

        wall_patch = self._extract_patch(wall_channel, next_pos).flatten()
        goal_patch = self._extract_patch(goal_channel, next_pos).flatten()

        features = np.concatenate(
            [
                np.array([1.0, invalid_move, next_is_goal, distance_delta, neighbor_ratio, is_dead_end], dtype=np.float32),
                wall_patch.astype(np.float32),
                goal_patch.astype(np.float32),
            ]
        )
        return features

    def _find_position(self, plane: np.ndarray, name: str) -> tuple[int, int]:
        coords = np.argwhere(plane > 0.5)
        if coords.size == 0:
            raise ValueError(f"Could not find {name} position in observation.")
        return tuple(coords[0].tolist())

    def _next_position(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        y, x = pos
        if action == 0:
            return y - 1, x
        if action == 1:
            return y, x + 1
        if action == 2:
            return y + 1, x
        if action == 3:
            return y, x - 1
        return y, x

    def _is_free(self, wall_plane: np.ndarray, pos: tuple[int, int]) -> bool:
        y, x = pos
        if y < 0 or y >= wall_plane.shape[0] or x < 0 or x >= wall_plane.shape[1]:
            return False
        return wall_plane[y, x] == 0.0

    def _manhattan_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _count_free_neighbors(self, wall_plane: np.ndarray, pos: tuple[int, int]) -> int:
        count = 0
        for neighbor in [
            (pos[0] - 1, pos[1]),
            (pos[0], pos[1] + 1),
            (pos[0] + 1, pos[1]),
            (pos[0], pos[1] - 1),
        ]:
            if self._is_free(wall_plane, neighbor):
                count += 1
        return count

    def _extract_patch(self, plane: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        patch = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
        cy, cx = center
        for dy in range(-self.patch_radius, self.patch_radius + 1):
            for dx in range(-self.patch_radius, self.patch_radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < plane.shape[0] and 0 <= nx < plane.shape[1]:
                    patch[dy + self.patch_radius, dx + self.patch_radius] = plane[ny, nx]
        return patch


class TDLambdaActionFeatureAgent(BaseAgent):
    """TD(λ) agent that learns explicit state-action features for navigation."""

    def __init__(
        self,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 5e-4,
        epsilon: float = 0.1,
        lambda_value: float = 0.9,
        patch_radius: int = 1,
        seed: int | None = None,
    ):
        self.feature_extractor = ActionConditionalFeatureExtractor(patch_radius=patch_radius)
        state_size = self.feature_extractor.feature_size
        super().__init__(state_size, n_actions, gamma, alpha, epsilon, seed)
        self.lambda_value = float(lambda_value)
        self.weights = np.zeros((n_actions, state_size), dtype=np.float32)
        self.eligibility = np.zeros_like(self.weights)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        values = np.zeros(self.n_actions, dtype=np.float32)
        for action in range(self.n_actions):
            features = self.feature_extractor.extract(state, action)
            values[action] = float(self.weights[action].dot(features))
        return values

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
        features = self.feature_extractor.extract(state, action)
        self.eligibility *= self.gamma * self.lambda_value
        self.eligibility[action] += features
        self.weights += self.alpha * td_error * self.eligibility

    def save(self, path: str) -> None:
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        self.weights = np.load(path)


__all__ = ["ActionConditionalFeatureExtractor", "TDLambdaActionFeatureAgent"]
