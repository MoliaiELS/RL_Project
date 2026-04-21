import numpy as np

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def get_valid_actions(agent_pos: tuple[int, int], index_map: np.ndarray) -> np.ndarray:
    """Return a boolean mask of valid actions for the current agent position."""
    valid_actions = np.zeros(len(DIRECTIONS), dtype=bool)
    height, width = index_map.shape
    y, x = agent_pos
    for action, (dy, dx) in enumerate(DIRECTIONS):
        ny = y + dy
        nx = x + dx
        if 0 <= ny < height and 0 <= nx < width and index_map[ny, nx] >= 0:
            valid_actions[action] = True
    return valid_actions


def masked_q_values(q_values: np.ndarray, valid_actions: np.ndarray) -> np.ndarray:
    """Mask invalid actions for action selection without changing the true network outputs."""
    q_masked = np.array(q_values, dtype=float, copy=True)
    if valid_actions.size != q_masked.size:
        raise ValueError("valid_actions mask and q_values must have same shape")
    if not np.any(valid_actions):
        return q_masked
    q_masked[~valid_actions] = float("-inf")
    return q_masked


def select_action_from_valid_actions(
    q_values: np.ndarray,
    valid_actions: np.ndarray,
    rng: np.random.Generator,
    greedy: bool = True,
) -> int:
    """Select either a greedy or random valid action given a valid-action mask."""
    if valid_actions.size != q_values.size:
        raise ValueError("valid_actions mask and q_values must have same shape")
    valid_indices = np.nonzero(valid_actions)[0]
    if valid_indices.size == 0:
        # Fallback to the full action space if no valid actions exist.
        return int(rng.integers(q_values.size))
    if greedy:
        masked = masked_q_values(q_values, valid_actions)
        max_value = np.max(masked)
        best_actions = np.flatnonzero(masked == max_value)
        return int(rng.choice(best_actions))
    return int(rng.choice(valid_indices))
