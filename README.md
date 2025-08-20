# 🚀 CTA Strategy Research and Backtesting Platform

A comprehensive cryptocurrency and stock factor research platform implementing CTA (Commodity Trading Advisor) strategies with advanced analytics, risk management, and optimization capabilities.

## Project Overview

This project implements a trend-following CTA strategy that:
- Trades both cryptocurrencies and stocks
- Uses Moving Average crossovers and RSI for signal generation
- Includes comprehensive risk management
- Provides detailed backtesting and performance analysis
- Features interactive visualization tools

## Strategy Description

The CTA strategy combines:
1. **Moving Average Crossovers**: Entry signals when fast MA crosses above slow MA
2. **RSI Filtering**: Additional confirmation using RSI levels (30/70 thresholds)
3. **Risk Management**: Stop-loss, position sizing, and drawdown controls
4. **Multi-Asset Support**: Simultaneous trading across crypto and stock markets

## Project Structure

```
crypto-cta-strategy/
├── data/                   # Data storage and cache
├── src/
│   ├── data/              # Data fetching and preprocessing
│   ├── indicators/        # Technical indicators implementation
│   ├── strategies/        # CTA strategy logic
│   ├── backtesting/       # Backtesting engine
│   ├── risk/              # Risk management modules
│   ├── analytics/         # Performance analysis
│   └── visualization/     # Plotting and reporting
├── notebooks/             # Jupyter notebooks for research
├── tests/                 # Unit tests
├── config/                # Configuration files
└── results/               # Backtest results and reports
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from src.strategies.cta_strategy import CTAStrategy
from src.backtesting.engine import BacktestEngine

# Initialize strategy
strategy = CTAStrategy(
    fast_ma_period=10,
    slow_ma_period=30,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70
)

# Run backtest
engine = BacktestEngine(strategy)
results = engine.run_backtest(['BTC-USD', 'AAPL'], start_date='2023-01-01')
```

## Features

- ✅ Multi-asset data fetching (crypto + stocks)
- ✅ Technical indicator library
- ✅ CTA strategy implementation
- ✅ Comprehensive backtesting engine
- ✅ Risk management system
- ✅ Performance analytics
- ✅ Interactive visualizations
- ✅ Parameter optimization
- ✅ Portfolio management

## License

MIT License
