from DataDownloader import YFinanceDownloader
from TradingEnv import TradingEnv
from DqnTradingAgent import DqnTradingAgent
from PerformanceMetrics import PerformanceMetrics
from FeatureEngineer import FeatureEngineer

import sys
sys.dont_write_bytecode = True


def Main():
    Downloader = YFinanceDownloader("AAPL", "2020-01-01", "2023-12-31", "1d")
    Data = Downloader.DownloadData()
    Engineer = FeatureEngineer(Data, IncludeIndicators=True)
    Data = Engineer.Transform()
    Environment = TradingEnv(
        DataFrame=Data,
        WindowSize=5,
        InitialBalance=1000,
        TransactionFee=0.001,
        SharpeRatioWeight=0.1,
        DrawdownWeight=0.1,
        VolatilityWeight=0.05,
    )
    Agent = DqnTradingAgent(
        Environment,
        LearningRate=1e-3,
        BufferSize=50000,
        BatchSize=64,
        Gamma=0.95,
    )
    Metrics = PerformanceMetrics(Environment)
    Agent.Train(Timesteps=1000)
    Observation, Info = Environment.Reset()
    Metrics.Record()
    Done = False
    while not Done:
        Action = Agent.Predict(Observation)
        Observation, Reward, Done, _, Info = Environment.Step(Action)
        Metrics.Record()

    Metrics.PlotAndPrint()


if __name__ == "__main__":
    Main()
