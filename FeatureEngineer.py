import pandas as pd
import numpy as np

# Ensure compatibility with pandas_ta on newer versions of numpy
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, DataFrame: pd.DataFrame, IncludeIndicators: bool = True):
        self.DataFrame = DataFrame.copy()
        self.IncludeIndicators = IncludeIndicators
        self.Indicators = {}
        if self.IncludeIndicators:
            # Register default indicators
            self.RegisterIndicator("SMA_10", self._SimpleMovingAverageFactory(10))
            self.RegisterIndicator("SMA_20", self._SimpleMovingAverageFactory(20))
            self.RegisterIndicator("EMA_10", self._ExponentialMovingAverageFactory(10))
            self.RegisterIndicator("MACD", self._MacdFactory())
            self.RegisterIndicator("RSI_14", self._RelativeStrengthIndexFactory(14))
            self.RegisterIndicator("BBANDS", self._BollingerBandsFactory())

    def RegisterIndicator(self, Name: str, Function):
        self.Indicators[Name] = Function

    def _SimpleMovingAverageFactory(self, Period: int):
        def CalculateSMA(DataFrame: pd.DataFrame):
            return ta.sma(DataFrame["Close"], length=Period)
        return CalculateSMA

    def _RelativeStrengthIndexFactory(self, Period: int):
        def CalculateRSI(DataFrame: pd.DataFrame):
            return ta.rsi(DataFrame["Close"], length=Period)
        return CalculateRSI

    def _ExponentialMovingAverageFactory(self, Period: int):
        def CalculateEMA(DataFrame: pd.DataFrame):
            return ta.ema(DataFrame["Close"], length=Period)
        return CalculateEMA

    def _MacdFactory(self, Fast: int = 12, Slow: int = 26, Signal: int = 9):
        def CalculateMACD(DataFrame: pd.DataFrame):
            MacdDf = ta.macd(DataFrame["Close"], fast=Fast, slow=Slow, signal=Signal)
            MacdDf.columns = ["MACD", "MACDh", "MACDs"]
            return MacdDf
        return CalculateMACD

    def _BollingerBandsFactory(self, Length: int = 20, Std: float = 2.0):
        def CalculateBollinger(DataFrame: pd.DataFrame):
            BbDf = ta.bbands(DataFrame["Close"], length=Length, std=Std)
            BbDf.columns = ["BBL", "BBM", "BBU", "BBB", "BBP"]
            return BbDf
        return CalculateBollinger

    def Transform(self):
        if self.IncludeIndicators:
            for Name, Function in self.Indicators.items():
                Result = Function(self.DataFrame)
                if isinstance(Result, pd.DataFrame):
                    for Column in Result.columns:
                        self.DataFrame[f"{Column}"] = Result[Column]
                else:
                    self.DataFrame[Name] = Result
        self.DataFrame.dropna(inplace=True)
        return self.DataFrame
