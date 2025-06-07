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
            self.RegisterIndicator("RSI_14", self._RelativeStrengthIndexFactory(14))

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

    def Transform(self):
        if self.IncludeIndicators:
            for Name, Function in self.Indicators.items():
                self.DataFrame[Name] = Function(self.DataFrame)
        self.DataFrame.dropna(inplace=True)
        return self.DataFrame
