import gymnasium as gym

try:
    from stable_baselines3 import PPO
except ImportError as err:
    raise ImportError(
        "stable-baselines3 is required for PPO training. Install it with `pip install stable-baselines3`."
    ) from err


class PPOAgent:
    def __init__(
        self,
        env_id: str,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        seed: int | None = None,
    ):
        self.env_id = env_id
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.model = None

    def build(self, env=None) -> None:
        if env is None:
            env = gym.make(self.env_id)
        self.model = PPO(
            "MlpPolicy",
            env,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            verbose=1,
            seed=self.seed,
        )

    def train(self, total_timesteps: int) -> None:
        if self.model is None:
            self.build()
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not built or trained yet.")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.predict(observation, deterministic=deterministic)


__all__ = ["PPOAgent"]
