import numpy as np
import pandas as pd
import gymnasium as gym # gymnasium is the updated version of gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.

    The environment simulates trading a single stock.
    The agent can choose to buy, sell, or hold.
    The reward is based on the change in portfolio value (total return).
    Transaction fees are incorporated into the reward calculation.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, DataFrame: pd.DataFrame, InitialBalance: float = 100000.0, TransactionFeePercent: float = 0.001, LookBackWindow: int = 30):
        super(TradingEnv, self).__init__()

        if DataFrame.empty:
            raise ValueError("DataFrame cannot be empty.")

        # DataFrame index validation and conversion
        if not isinstance(DataFrame.index, pd.DatetimeIndex):
            original_index_name = DataFrame.index.name
            # Attempt to convert if it's a default RangeIndex or if name suggests it's a date/datetime
            if isinstance(DataFrame.index, pd.RangeIndex) or original_index_name in ['Datetime', 'Date', 'datetime', 'date']:
                try:
                    DataFrame.index = pd.to_datetime(DataFrame.index)
                    if original_index_name: # Preserve original name if it existed
                       DataFrame.index.name = original_index_name
                    else: # Default to 'Datetime' if no name was present
                       DataFrame.index.name = 'Datetime'
                    print(f"Converted DataFrame index to DatetimeIndex. Index name: {DataFrame.index.name}")
                except Exception as e:
                    raise ValueError(f"DataFrame index was of type {type(DataFrame.index)} (name: {original_index_name}) and could not be converted to DatetimeIndex: {e}")
            else: # Final check if not convertible
                 raise ValueError(f"DataFrame index must be a DatetimeIndex. Got {type(DataFrame.index)} with name {original_index_name}.")


        if InitialBalance <= 0:
            raise ValueError("InitialBalance must be positive.")
        if TransactionFeePercent < 0:
            raise ValueError("TransactionFeePercent cannot be negative.")
        if LookBackWindow <= 0:
            raise ValueError("LookBackWindow must be positive.")
        if len(DataFrame) <= LookBackWindow:
            raise ValueError(f"DataFrame length ({len(DataFrame)}) must be greater than LookBackWindow ({LookBackWindow}).")


        self.DataFrame = DataFrame.copy() # Data with features
        self.InitialBalance = InitialBalance
        self.TransactionFeePercent = TransactionFeePercent
        self.LookBackWindow = LookBackWindow # Number of previous time steps to include in observation

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Ensure 'Close' column exists for price information
        if 'Close' not in self.DataFrame.columns:
            raise ValueError("DataFrame must contain a 'Close' column for price information.")

        ObservationFeatures = self.DataFrame.select_dtypes(include=np.number).columns.tolist()
        if not ObservationFeatures:
            raise ValueError("DataFrame contains no numeric columns to use as features for observation.")

        self.ObservationShape = (LookBackWindow, len(ObservationFeatures))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.ObservationShape, dtype=np.float32)

        self.CurrentStep = 0
        self.Balance = 0.0
        self.SharesHeld = 0
        self.NetWorth = 0.0
        self.TradeHistory = []

        # self.reset() # Initialize state variables - reset is called by the user/runner

    def _GetObservation(self):
        # Observation is data from (CurrentStep - LookBackWindow) up to (but not including) CurrentStep
        Observation = self.DataFrame.iloc[self.CurrentStep - self.LookBackWindow : self.CurrentStep].select_dtypes(include=np.number).values
        return np.array(Observation, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.CurrentStep = self.LookBackWindow
        self.Balance = self.InitialBalance
        self.SharesHeld = 0
        self.NetWorth = self.InitialBalance
        self.TradeHistory = []

        Observation = self._GetObservation()
        Info = self._GetInfo()
        return Observation, Info

    def _GetInfo(self):
        # Price for info is the price at CurrentStep-1 (last state) or start if CurrentStep is LookBackWindow
        PriceIndex = max(self.LookBackWindow, self.CurrentStep -1) # Ensure it's a valid index past lookback
        PriceIndex = min(PriceIndex, len(self.DataFrame) - 1)

        return {
            "CurrentStep": self.CurrentStep,
            "Balance": self.Balance,
            "SharesHeld": self.SharesHeld,
            "NetWorth": self.NetWorth,
            "CurrentPrice": self.DataFrame['Close'].iloc[PriceIndex], # Price of the state returned by reset or step
            "TradeHistorySize": len(self.TradeHistory)
        }

    def step(self, Action):
        # The price for action is at CurrentStep (the state we are currently in before moving to next)
        CurrentPrice = self.DataFrame['Close'].iloc[self.CurrentStep]
        PreviousNetWorth = self.NetWorth

        if Action == 1: # Buy
            SharesToBuy = (self.Balance / (CurrentPrice * (1 + self.TransactionFeePercent)))
            SharesToBuy = np.floor(SharesToBuy) # Whole shares

            if SharesToBuy > 0:
                Cost = SharesToBuy * CurrentPrice
                Fee = Cost * self.TransactionFeePercent
                self.Balance -= (Cost + Fee)
                self.SharesHeld += SharesToBuy
                self.TradeHistory.append({"Step": self.CurrentStep, "Action": "BUY", "Price": CurrentPrice, "Shares": SharesToBuy, "Fee": Fee})
        elif Action == 2: # Sell
            if self.SharesHeld > 0:
                Revenue = self.SharesHeld * CurrentPrice
                Fee = Revenue * self.TransactionFeePercent
                self.Balance += (Revenue - Fee)
                self.TradeHistory.append({"Step": self.CurrentStep, "Action": "SELL", "Price": CurrentPrice, "Shares": self.SharesHeld, "Fee": Fee})
                self.SharesHeld = 0

        # Move to the next step in time
        self.CurrentStep += 1

        # Update NetWorth: Balance + value of shares held at the new CurrentStep's closing price
        # (or the last available price if CurrentStep is now beyond data)
        NextPriceIndex = min(self.CurrentStep, len(self.DataFrame) - 1)
        self.NetWorth = self.Balance + (self.SharesHeld * self.DataFrame['Close'].iloc[NextPriceIndex])

        Reward = self.NetWorth - PreviousNetWorth

        # Done if CurrentStep reaches a point where the next observation would be incomplete or out of bounds.
        # The agent acts at CurrentStep. The next observation will be for CurrentStep+1.
        # This observation requires data from (CurrentStep+1 - LookBackWindow) to (CurrentStep+1).
        # Thus, the last valid CurrentStep for providing a *full* next observation is when (CurrentStep+1) == len(DataFrame).
        # So, if CurrentStep reaches len(DataFrame), the episode ends.
        Done = self.CurrentStep >= len(self.DataFrame)

        Observation = self._GetObservation() if not Done else np.zeros(self.ObservationShape, dtype=np.float32)
        Info = self._GetInfo() # Info should reflect the state *after* the action
        Truncated = False

        return Observation, Reward, Done, Truncated, Info

    def render(self, mode='human'):
        if mode == 'human':
            # Price for render should be the price related to the current state (after action)
            PriceIndexRender = min(self.CurrentStep, len(self.DataFrame) -1)
            print(f"Step: {self.CurrentStep}")
            print(f"Balance: {self.Balance:.2f}")
            print(f"Shares Held: {self.SharesHeld}")
            print(f"Net Worth: {self.NetWorth:.2f} (Price at Render: {self.DataFrame['Close'].iloc[PriceIndexRender]:.2f})")
            LastTrade = self.TradeHistory[-1] if self.TradeHistory else "None"
            print(f"Last Trade: {LastTrade}")
            print("-" * 30)

    def close(self):
        print("Trading environment closed.")


if __name__ == '__main__':
    FeatureData = None
    try:
        from DataDownloader import DownloadData
        from FeatureEngineer import AddFeatures

        print("Attempting to download and feature engineer data for AAPL...")
        RawData = DownloadData(Ticker="AAPL", StartDate="2020-01-01", EndDate="2023-12-31", Interval="1d")

        if RawData.empty:
            print("Failed to download AAPL data. Falling back to dummy data.")
        else:
            FeatureData = AddFeatures(RawData.copy())
            if FeatureData.empty or len(FeatureData) < 60: # Need enough data for lookback + steps
                 print(f"Not enough data after feature engineering (got {len(FeatureData)} rows). Falling back to dummy data.")
                 FeatureData = None # Force fallback
            else:
                print(f"Data ready. Shape: {FeatureData.shape}")

    except Exception as E:
        print(f"Could not load or process real data due to: {E}. Using dummy data for TradingEnv example.")
        FeatureData = None # Ensure fallback

    if FeatureData is None: # Fallback to dummy data
        print("Using dummy data for TradingEnv example.")
        NumRows = 200
        Dates = pd.date_range(start='2022-01-01', periods=NumRows, freq='B')
        FeatureData = pd.DataFrame(index=Dates)
        FeatureData.index.name = 'Datetime'

        # Base 'Close' prices with some trend and noise
        BasePrice = 150
        Trend = np.arange(NumRows) * 0.1
        Noise = np.random.normal(loc=0, scale=5, size=NumRows)
        FeatureData['Close'] = (BasePrice + Trend + Noise).round(2)
        FeatureData['Close'] = np.maximum(FeatureData['Close'], 1) # Ensure price is positive

        # Generate other OHLC based on Close
        FeatureData['Open'] = (FeatureData['Close'] - np.random.normal(loc=0, scale=1, size=NumRows)).round(2)
        FeatureData['High'] = (FeatureData['Close'] + np.random.uniform(low=0, high=3, size=NumRows)).round(2)
        FeatureData['Low'] = (FeatureData['Close'] - np.random.uniform(low=0, high=3, size=NumRows)).round(2)
        FeatureData['High'] = np.maximum(FeatureData['High'], FeatureData[['Open', 'Close']].max(axis=1))
        FeatureData['Low'] = np.minimum(FeatureData['Low'], FeatureData[['Open', 'Close']].min(axis=1))
        FeatureData['Low'] = np.maximum(FeatureData['Low'], 0.5) # Ensure low is positive

        FeatureData['Volume'] = np.abs(np.random.normal(loc=1000000, scale=300000, size=NumRows)).round(0) + 100000

        # Dummy technical indicators
        FeatureData['SMA50'] = FeatureData['Close'].rolling(window=min(50, NumRows//4)).mean().fillna(method='bfill').fillna(method='ffill')
        FeatureData['SMA200'] = FeatureData['Close'].rolling(window=min(200, NumRows//2)).mean().fillna(method='bfill').fillna(method='ffill')
        FeatureData['RSI'] = np.random.normal(loc=50, scale=10, size=NumRows).round(2)
        FeatureData['MACD'] = np.random.normal(loc=0, scale=1, size=NumRows).round(4)
        FeatureData['MACDSignal'] = np.random.normal(loc=0, scale=1, size=NumRows).round(4)
        FeatureData['MACDDiff'] = FeatureData['MACD'] - FeatureData['MACDSignal']
        FeatureData['BollingerHigh'] = FeatureData['SMA50'] + 2 * FeatureData['Close'].rolling(window=min(20, NumRows//5)).std().fillna(method='bfill').fillna(method='ffill')
        FeatureData['BollingerLow'] = FeatureData['SMA50'] - 2 * FeatureData['Close'].rolling(window=min(20, NumRows//5)).std().fillna(method='bfill').fillna(method='ffill')
        FeatureData['BollingerMid'] = FeatureData['SMA50']
        # Ensure all features are present as in AddFeatures for consistency
        FeatureData['Adj Close'] = FeatureData['Close'] # Often same as Close for adjusted data

        # Fill any remaining NaNs from rolling functions at the start
        FeatureData.fillna(method='bfill', inplace=True)
        FeatureData.fillna(method='ffill', inplace=True)
        print(f"Dummy data generated. Shape: {FeatureData.shape}")


    LookBack = 20
    if len(FeatureData) <= LookBack:
        print(f"Error: FeatureData length ({len(FeatureData)}) must be greater than LookBackWindow ({LookBack}). Exiting.")
        # exit() # Avoid exiting in automated environment, let it fail if it must.
        # For now, if this condition is met, the Env init will raise ValueError.

    try:
        Env = TradingEnv(DataFrame=FeatureData, InitialBalance=10000, TransactionFeePercent=0.001, LookBackWindow=LookBack)

        Obs, Info = Env.reset()
        print("\nEnvironment Reset:")
        print(f"Initial Observation Shape: {Obs.shape}")
        # print(f"Initial Observation Sample (first step):\n{Obs[0]}") # Print first part of lookback
        print(f"Initial Info: {Info}")
        Env.render()

        TotalReward = 0.0
        MaxSteps = len(FeatureData) - LookBack -1 # Max steps agent can take. Loop runs 0 to MaxSteps-1
                                                 # CurrentStep goes from LookBack to len(FeatureData)-1
                                                 # So, number of steps is len(FeatureData)-1 - LookBack + 1 = len(FeatureData) - LookBack

        print(f"Max possible steps in this episode: {MaxSteps +1 }") # Corrected max steps calculation

        for i in range(MaxSteps + 1):
            Action = Env.action_space.sample()
            Obs, Reward, Done, Truncated, Info = Env.step(Action)
            TotalReward += Reward
            # print(f"Action Taken: {['Hold', 'Buy', 'Sell'][Action]} at time step {Env.CurrentStep-1}") # Action was for CurrentStep-1 state
            Env.render()
            if Done:
                print(f"Episode finished after {i+1} steps.")
                break

        if not Done: # Should be done if loop completes fully
             print(f"Episode loop completed {i+1} steps. Final state may not be 'Done' if MaxSteps logic is off.")

        print(f"\nTotal Reward from random actions: {TotalReward:.2f}")
        Env.close()

    except ValueError as VE:
        print(f"\nError creating or running TradingEnv: {VE}")
    except Exception as E:
        print(f"\nAn unexpected error occurred: {E}")
