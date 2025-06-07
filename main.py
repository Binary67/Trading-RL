from DataDownloader import YFinanceDownloader
from TradingEnv import TradingEnv
from DqnTradingAgent import DqnTradingAgent


def Main():
    Downloader = YFinanceDownloader("AAPL", "2020-01-01", "2023-12-31", "1d")
    Data = Downloader.DownloadData()
    Environment = TradingEnv(DataFrame=Data, WindowSize=5, InitialBalance=1000)
    Agent = DqnTradingAgent(Environment)
    Agent.Train(Timesteps=1000)
    Observation, Info = Environment.Reset()
    Done = False
    while not Done:
        Action = Agent.Predict(Observation)
        Observation, Reward, Done, _, Info = Environment.Step(Action)


if __name__ == "__main__":
    Main()
