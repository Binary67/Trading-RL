from stable_baselines3 import DQN

class DqnTradingAgent:
    def __init__(
        self,
        Environment,
        LearningRate: float = 1e-4,
        BufferSize: int = 100000,
        BatchSize: int = 32,
        Gamma: float = 0.99,
    ):
        self.Environment = Environment
        self.Model = DQN(
            "MlpPolicy",
            self.Environment,
            learning_rate=LearningRate,
            buffer_size=BufferSize,
            batch_size=BatchSize,
            gamma=Gamma,
            verbose=0,
        )

    def Train(self, Timesteps: int = 1000):
        self.Model.learn(total_timesteps=Timesteps)

    def Predict(self, Observation):
        Action, _ = self.Model.predict(Observation)
        return Action
