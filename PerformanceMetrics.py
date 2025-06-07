import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PerformanceMetrics:
    def __init__(self, Environment):
        self.Env = Environment
        self.EquityCurve = []

    def Record(self):
        Price = float(self.Env.DataFrame.loc[self.Env.CurrentStep, "Close"])
        Equity = self.Env.CurrentBalance + self.Env.SharesHeld * Price
        self.EquityCurve.append(Equity)

    def PlotAndPrint(self):
        EquitySeries = pd.Series(self.EquityCurve)
        Returns = EquitySeries.pct_change().dropna()
        if len(Returns) > 0 and Returns.std() != 0:
            SharpeRatio = np.sqrt(252) * Returns.mean() / Returns.std()
        else:
            SharpeRatio = np.nan
        Drawdown = 1 - EquitySeries / EquitySeries.cummax()
        MaxDrawdown = Drawdown.max()
        CumulativeReturn = (EquitySeries.iloc[-1] / EquitySeries.iloc[0]) - 1
        plt.figure()
        plt.plot(EquitySeries)
        plt.title("Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.show()
        print(f"Cumulative Return: {CumulativeReturn:.2%}")
        print(f"Sharpe Ratio: {SharpeRatio:.2f}")
        print(f"Max Drawdown: {MaxDrawdown:.2%}")
