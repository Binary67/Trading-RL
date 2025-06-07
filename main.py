from DataDownloader import YFinanceDownloader
from TradingEnv import TradingEnv


def Main():
    Downloader = YFinanceDownloader("AAPL", "2023-01-01", "2023-01-15", "1d")
    Data = Downloader.DownloadData()
    Environment = TradingEnv(DataFrame=Data, WindowSize=5, InitialBalance=1000)
    Observation, Info = Environment.Reset()
    Done = False
    while not Done:
        Action = Environment.action_space.sample()
        Observation, Reward, Done, _, Info = Environment.Step(Action)


if __name__ == "__main__":
    Main()
