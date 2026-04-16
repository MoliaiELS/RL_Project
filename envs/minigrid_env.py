import numpy as np

try:
    import gymnasium as gym
    import minigrid.envs  # noqa: F401
    from minigrid.wrappers import FlatObsWrapper
    _USE_GYM = False
except ImportError:
    import gym
    import gym_minigrid.envs  # noqa: F401
    from gym_minigrid.wrappers import FlatObsWrapper
    _USE_GYM = True

from .maze_env import make_maze_env


def make_minigrid_env(env_id: str = "MiniGrid-Empty-8x8-v0", seed: int | None = None):
    try:
        env = gym.make(env_id)
    except Exception:
        if _USE_GYM:
            raise
        from gym import make as gym_make
        env = gym_make(env_id)
    env = FlatObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env(env_id: str = "MiniGrid-Empty-8x8-v0", seed: int | None = None, use_manhattan_distance: bool = True):
    if env_id.startswith("Maze-"):
        return make_maze_env(env_id, seed=seed, use_manhattan_distance=use_manhattan_distance)
    return make_minigrid_env(env_id, seed=seed)


class MiniGridEncoder:
    def __init__(self, observation_space: gym.Space):
        self.shape = observation_space.shape
        self.size = int(np.prod(self.shape))

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim > 1:
            x = x.flatten()
        if x.size == 0:
            return x
        max_value = float(x.max())
        if max_value > 0.0:
            x = x / max_value
        return x


__all__ = ["make_minigrid_env", "make_env", "MiniGridEncoder"]
