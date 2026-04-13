import gymnasium as gym
import numpy as np
from gym_minigrid.wrappers import FlatObsWrapper


def make_minigrid_env(env_id: str = "MiniGrid-Empty-8x8-v0", seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


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


__all__ = ["make_minigrid_env", "MiniGridEncoder"]
