import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(
        self,
        DataFrame: pd.DataFrame,
        WindowSize: int = 10,
        InitialBalance: float = 1000.0,
        TransactionFee: float = 0.0,
        SharpeRatioWeight: float = 0.0,
    ):
        super().__init__()
        self.DataFrame = DataFrame.reset_index(drop=True)
        self.WindowSize = WindowSize
        self.InitialBalance = InitialBalance
        self.TransactionFee = TransactionFee
        self.SharpeRatioWeight = SharpeRatioWeight
        self.CurrentBalance = self.InitialBalance
        self.SharesHeld = 0
        self.CurrentStep = self.WindowSize
        self.ReturnsHistory = []
        self.ActionSpace = spaces.Discrete(3)
        self.ObservationSpace = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.WindowSize, self.DataFrame.shape[1]),
            dtype=np.float32
        )
        self.action_space = self.ActionSpace
        self.observation_space = self.ObservationSpace

    def _GetObservation(self):
        Frame = self.DataFrame.iloc[self.CurrentStep - self.WindowSize:self.CurrentStep]
        return Frame.values.astype(np.float32)

    def Reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.CurrentBalance = self.InitialBalance
        self.SharesHeld = 0
        self.CurrentStep = self.WindowSize
        self.ReturnsHistory = []
        Observation = self._GetObservation()
        Info = {"balance": self.CurrentBalance, "shares_held": self.SharesHeld}
        return Observation, Info

    # Compatibility methods for libraries expecting lower-case names
    def reset(self, seed=None, options=None):
        return self.Reset(seed=seed, options=options)

    def Step(self, Action: int):
        Price = float(self.DataFrame.loc[self.CurrentStep, "Close"])
        PortfolioValue = self.CurrentBalance + self.SharesHeld * Price
        TransactionCost = 0.0
        if Action == 0:
            # Hold action: do nothing
            pass
        elif Action == 1:
            SharesToBuy = int(self.CurrentBalance // Price)
            self.CurrentBalance -= SharesToBuy * Price
            TransactionCost = SharesToBuy * Price * self.TransactionFee
            self.CurrentBalance -= TransactionCost
            self.SharesHeld += SharesToBuy
        elif Action == 2:
            TransactionCost = self.SharesHeld * Price * self.TransactionFee
            self.CurrentBalance += self.SharesHeld * Price
            self.CurrentBalance -= TransactionCost
            self.SharesHeld = 0
        self.CurrentStep += 1
        if self.CurrentStep >= len(self.DataFrame):
            self.CurrentStep = len(self.DataFrame) - 1
            Terminated = True
        else:
            Terminated = False
        NextPrice = float(self.DataFrame.loc[self.CurrentStep, "Close"])
        NextValue = self.CurrentBalance + self.SharesHeld * NextPrice
        ValueChange = NextValue - PortfolioValue
        if PortfolioValue != 0:
            Return = ValueChange / PortfolioValue
        else:
            Return = 0.0
        self.ReturnsHistory.append(Return)
        if len(self.ReturnsHistory) > 1 and np.std(self.ReturnsHistory) != 0:
            SharpeRatio = np.sqrt(252) * np.mean(self.ReturnsHistory) / np.std(self.ReturnsHistory)
        else:
            SharpeRatio = 0.0
        Reward = ValueChange - TransactionCost + self.SharpeRatioWeight * SharpeRatio
        Observation = self._GetObservation()
        Info = {"balance": self.CurrentBalance, "shares_held": self.SharesHeld}
        return Observation, Reward, Terminated, False, Info

    # Compatibility method for libraries expecting lower-case names
    def step(self, Action: int):
        return self.Step(Action)
