from stable_baselines3 import DQN

class DqnTradingAgent:
    def __init__(self, Environment):
        self.Environment = Environment
        self.Model = DQN("MlpPolicy", self.Environment, verbose=0)

    def Train(self, Timesteps: int = 1000):
        self.Model.learn(total_timesteps=Timesteps)

    def Predict(self, Observation):
        Action, _ = self.Model.predict(Observation)
        return Action
