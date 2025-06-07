# Trading-RL

Trading-RL provides a simple framework for training a reinforcement learning agent on stock market data. The project demonstrates how to download market data, engineer technical indicators and train a Deep Q-Network (DQN) agent to manage a trading portfolio.

## Features

- **DataDownloader** – fetches historical OHLCV data from Yahoo Finance and automatically handles hourly download limits.
- **FeatureEngineer** – computes common technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands) and supports `z-score` or `min-max` normalization.
- **TradingEnv** – a `gymnasium` environment that simulates trading with transaction fees, slippage and risk-adjusted rewards.
- **DqnTradingAgent** – wraps a Stable Baselines3 `DQN` model for training and inference.
- **PerformanceMetrics** – records the agent's equity curve and prints cumulative return, Sharpe ratio and maximum drawdown.

## Installation

Install the required packages using pip:

```bash
pip install stable-baselines3 gymnasium pandas pandas-ta yfinance matplotlib
```

## Usage

Run the included example script:

```bash
python main.py
```

The example downloads Apple price data, generates features, trains a DQN agent for a short period and then evaluates its performance. An `equity_curve.png` file will be produced alongside printed statistics.

## Repository Structure

- `DataDownloader.py` – data acquisition utilities.
- `FeatureEngineer.py` – feature calculation and normalization.
- `TradingEnv.py` – the custom trading environment.
- `DqnTradingAgent.py` – the reinforcement learning agent implementation.
- `PerformanceMetrics.py` – evaluation and plotting helpers.
- `main.py` – demonstrates the complete workflow.

The code is intended for educational use and can be extended for more advanced research.
