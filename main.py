from DataDownloader import YFinanceDownloader
from TradingEnv import TradingEnv
from DqnTradingAgent import DqnTradingAgent
from PerformanceMetrics import PerformanceMetrics


def Main():
    Downloader = YFinanceDownloader("AAPL", "2020-01-01", "2023-12-31", "1d")
    Data = Downloader.DownloadData()
    Environment = TradingEnv(DataFrame=Data, WindowSize=5, InitialBalance=1000)
    Agent = DqnTradingAgent(Environment)
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
