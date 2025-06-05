import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG # Import common algorithms
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback # For custom callbacks if needed
import gymnasium as gym # Ensure gymnasium is imported for type checking if needed

# Attempt to import custom modules
try:
    from DataDownloader import DownloadData
    from FeatureEngineer import AddFeatures
    from TradingEnvironment import TradingEnv
except ImportError as E:
    print(f"Error importing custom modules: {E}. Make sure they are in the PYTHONPATH or same directory.")
    # Define dummy classes/functions if imports fail, to allow script to be parsed
    if "DataDownloader" in str(E):
        def DownloadData(Ticker, StartDate, EndDate, Interval): return pd.DataFrame()
    if "FeatureEngineer" in str(E):
        def AddFeatures(DataFrame): return DataFrame
    if "TradingEnvironment" in str(E): # Placeholder if TradingEnv is missing
        class TradingEnv(gym.Env):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.action_space = gym.spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,10), dtype=np.float32)
                self.InitialBalance = 10000 # Dummy value for evaluation print
            def reset(self, seed=None, options=None): return np.zeros((10,10), dtype=np.float32), {}
            def step(self, action): return np.zeros((10,10), dtype=np.float32), 0, True, False, {'NetWorth': self.InitialBalance}
            def render(self): pass
            def close(self): pass


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            # Check if the environment is a VecEnv
            is_vec_env = hasattr(self.training_env, 'envs')

            if is_vec_env:
                # For VecEnvs, self.training_env.get_attr('attribute_name') returns a list
                # We typically log the value from the first environment for simplicity
                CurrentNetWorth = self.training_env.get_attr('NetWorth')[0]
                self.logger.record('custom/CurrentNetWorth', CurrentNetWorth)

                CurrentBalance = self.training_env.get_attr('Balance')[0]
                self.logger.record('custom/CurrentBalance', CurrentBalance)

                SharesHeld = self.training_env.get_attr('SharesHeld')[0]
                self.logger.record('custom/SharesHeld', SharesHeld)
            else:
                # For a single environment (not wrapped in VecEnv)
                # Access attributes directly if they exist, or via info dict if available through training_env
                # This part might need adjustment based on how SB3 wraps the single env
                # For simplicity, let's assume direct access or through a method if info dict is preferred.
                # This might be tricky as SB3 wraps the env, direct access to custom attributes is not straightforward
                # The most reliable way for single envs is usually through the info dict returned by step()
                # However, callback has access to `self.locals` which contains `infos`
                if self.locals.get("infos"):
                    info = self.locals["infos"][0] # Assuming single env
                    if "NetWorth" in info:
                        self.logger.record('custom/CurrentNetWorth', info["NetWorth"])
                    if "Balance" in info:
                        self.logger.record('custom/CurrentBalance', info["Balance"])
                    if "SharesHeld" in info:
                        self.logger.record('custom/SharesHeld', info["SharesHeld"])

        except Exception as e:
            if self.verbose > 0:
                print(f"TensorboardCallback: Error logging custom metrics - {e}")
            pass
        return True


def TrainAgent(Environment: TradingEnv, Algorithm: str = "PPO", TotalTimesteps: int = 10000, TensorboardLogPath: str = "./tensorboard_logs/"):
    """
    Trains a reinforcement learning agent on the given trading environment.
    """
    try:
        print("Checking environment compatibility...")
        check_env(Environment, warn=True)
        print("Environment check passed.")
    except Exception as E:
        print(f"Environment check failed: {E}")
        print("Please ensure the environment follows the Gym API.")
        return None

    os.makedirs(TensorboardLogPath, exist_ok=True)
    LogName = f"{Algorithm.upper()}_TradingAgent" # Ensure Algorithm is upper for consistent naming

    Model = None
    AlgorithmMap = {
        "PPO": PPO,
        "A2C": A2C,
        "DDPG": DDPG
    }

    if Algorithm.upper() in AlgorithmMap:
        SelectedAlgorithm = AlgorithmMap[Algorithm.upper()]

        if Algorithm.upper() == "DDPG" and isinstance(Environment.action_space, gym.spaces.Discrete):
             print(f"Warning: {Algorithm.upper()} is typically used with continuous action spaces. The current environment has a Discrete action space. SB3 may raise an error.")

        # Note: Some SB3 policies might not support complex observation spaces without CNNs (e.g. MultiInputPolicy for dict observations)
        # Our observation space is Box, so MlpPolicy should generally work.
        Model = SelectedAlgorithm("MlpPolicy", Environment, verbose=1, tensorboard_log=TensorboardLogPath)
        print(f"Training {Algorithm.upper()} agent for {TotalTimesteps} timesteps...")
        try:
            Model.learn(total_timesteps=TotalTimesteps, tb_log_name=LogName, callback=TensorboardCallback(verbose=1))
            print("Training complete.")
            ModelPath = os.path.join(TensorboardLogPath, f"{LogName}_final.zip")
            Model.save(ModelPath)
            print(f"Model saved as {ModelPath}")
        except Exception as E:
            print(f"An error occurred during training or saving: {E}")
            return None
    else:
        print(f"Algorithm {Algorithm} not supported. Choose from PPO, A2C, DDPG.")
        return None

    return Model

def EvaluateAgent(Environment: TradingEnv, Model, NumEpisodes: int = 10):
    """
    Evaluates a trained RL agent on the trading environment.
    """
    if Model is None or Environment is None:
        print("Model or Environment not provided for evaluation.")
        return

    print(f"\nEvaluating agent for {NumEpisodes} episodes...")
    TotalNetWorthAtEnd = 0
    TotalRewards = 0

    for Episode in range(NumEpisodes):
        Obs, Info = Environment.reset()
        Done = False
        EpisodeRewards = 0
        print(f"--- Episode {Episode + 1} ---")
        # Environment.render()
        while not Done:
            Action, _states = Model.predict(Obs, deterministic=True)
            Obs, Reward, Done, Truncated, Info = Environment.step(Action)
            EpisodeRewards += Reward
            # Environment.render()
            if Done:
                EpisodeNetWorth = Info.get('NetWorth', Environment.InitialBalance)
                print(f"Episode finished. Net Worth: {EpisodeNetWorth:.2f}, Total Reward for episode: {EpisodeRewards:.2f}")
                TotalNetWorthAtEnd += EpisodeNetWorth
                TotalRewards += EpisodeRewards
                break # Important to break here after Done

    AverageNetWorth = TotalNetWorthAtEnd / NumEpisodes
    AverageReward = TotalRewards / NumEpisodes
    print(f"\n--- Evaluation Summary ---")
    print(f"Average Net Worth after {NumEpisodes} episodes: {AverageNetWorth:.2f}")
    print(f"Average Reward per episode: {AverageReward:.2f}")
    InitialBalance = getattr(Environment, 'InitialBalance', 0) # Robustly get InitialBalance
    if InitialBalance > 0 :
        PercentReturn = ((AverageNetWorth - InitialBalance) / InitialBalance) * 100
        print(f"Average Percentage Return (based on Net Worth vs Initial Balance): {PercentReturn:.2f}%")
    else:
        print("Cannot calculate percentage return as InitialBalance is zero or not available.")


if __name__ == '__main__':
    # --- Configuration ---
    TickerSymbol = "MSFT"
    StartDate = "2022-01-01" # Shorter period for faster example run
    EndDate = "2023-12-31"
    Interval = "1d"
    LookBackWindowSize = 20 # Reduced for faster run, ensure > some minimums for indicators
    InitialAgentBalance = 50000
    TransactionFee = 0.001 # 0.1%

    AlgorithmToUse = "PPO"
    TrainingTimesteps = 15000 # Reduced for faster example, real training needs much more

    # --- 1. Load and Prepare Data ---
    print("Loading and preparing data...")
    HistoricalData = pd.DataFrame() # Initialize to ensure it exists
    FeatureEngineeredData = pd.DataFrame()

    try:
        HistoricalData = DownloadData(Ticker=TickerSymbol, StartDate=StartDate, EndDate=EndDate, Interval=Interval)
    except Exception as e:
        print(f"Error during DownloadData: {e}")

    if HistoricalData.empty:
        print(f"Failed to download data for {TickerSymbol}. Exiting.")
        exit()

    try:
        FeatureEngineeredData = AddFeatures(HistoricalData.copy())
    except Exception as e:
        print(f"Error during AddFeatures: {e}")

    if FeatureEngineeredData.empty or len(FeatureEngineeredData) <= LookBackWindowSize:
        print(f"Not enough data after feature engineering for Ticker {TickerSymbol} (need > {LookBackWindowSize} rows). Shape: {FeatureEngineeredData.shape}. Exiting.")
        exit()

    print(f"Data prepared. Shape: {FeatureEngineeredData.shape}")

    # --- 2. Setup Trading Environment ---
    print("\nSetting up trading environment...")
    TradeEnv = None
    try:
        TradeEnv = TradingEnv(
            DataFrame=FeatureEngineeredData,
            InitialBalance=InitialAgentBalance,
            TransactionFeePercent=TransactionFee,
            LookBackWindow=LookBackWindowSize
        )
        print("Trading environment setup complete.")
    except Exception as e:
        print(f"Error setting up TradingEnv: {e}")
        exit()

    # --- 3. Train Agent ---
    print("\nTraining agent...")
    TensorboardLogDirectory = f"./tensorboard_logs/{TickerSymbol}/"
    TrainedModel = TrainAgent(
        Environment=TradeEnv,
        Algorithm=AlgorithmToUse,
        TotalTimesteps=TrainingTimesteps,
        TensorboardLogPath=TensorboardLogDirectory
    )

    # --- 4. Evaluate Agent ---
    if TrainedModel:
        print("\nRe-initializing environment for evaluation...")
        EvalEnv = None
        try:
            EvalEnv = TradingEnv(
                DataFrame=FeatureEngineeredData,
                InitialBalance=InitialAgentBalance,
                TransactionFeePercent=TransactionFee,
                LookBackWindow=LookBackWindowSize
            )
            EvaluateAgent(Environment=EvalEnv, Model=TrainedModel, NumEpisodes=5)
        except Exception as e:
            print(f"Error during evaluation setup or run: {e}")
    else:
        print("Agent training failed or was skipped. No evaluation will be performed.")

    print("\nRLAgent.py script finished.")
