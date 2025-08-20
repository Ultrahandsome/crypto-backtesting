"""
Configuration settings for the CTA strategy platform.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Data sources configuration
DATA_CONFIG = {
    "crypto": {
        "exchange": "binance",
        "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"],
        "timeframes": ["1h", "4h", "1d"],
        "default_timeframe": "1h"
    },
    "stocks": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"],
        "timeframes": ["1h", "1d"],
        "default_timeframe": "1d"
    }
}

# Strategy default parameters
STRATEGY_CONFIG = {
    "cta": {
        "fast_ma_period": 10,
        "slow_ma_period": 30,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stop_loss_pct": 0.02,  # 2% stop loss
        "take_profit_pct": 0.06,  # 6% take profit
        "position_size": 0.1  # 10% of portfolio per trade
    }
}

# Backtesting configuration
BACKTEST_CONFIG = {
    "initial_capital": 100000,  # $100k starting capital
    "commission": 0.001,  # 0.1% commission
    "slippage": 0.0005,  # 0.05% slippage
    "start_date": "2023-01-01",
    "end_date": None  # None means current date
}

# Risk management settings
RISK_CONFIG = {
    "max_drawdown": 0.15,  # 15% maximum drawdown
    "max_positions": 5,  # Maximum concurrent positions
    "volatility_lookback": 20,  # Days for volatility calculation
    "risk_free_rate": 0.02  # 2% annual risk-free rate
}

# Visualization settings
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "style": "seaborn-v0_8",
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff7f0e"
    }
}

# API keys (to be set via environment variables)
API_KEYS = {
    "binance_api_key": os.getenv("BINANCE_API_KEY"),
    "binance_secret_key": os.getenv("BINANCE_SECRET_KEY"),
    "alpha_vantage_key": os.getenv("ALPHA_VANTAGE_KEY")
}
