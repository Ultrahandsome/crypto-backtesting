"""
Technical indicators module.
"""
from .base_indicator import BaseIndicator, MovingAverageBase, OscillatorBase
from .moving_averages import (
    SimpleMovingAverage, ExponentialMovingAverage, WeightedMovingAverage,
    MovingAverageCrossover, sma, ema, wma, ma_crossover
)
from .oscillators import (
    RSI, Stochastic, MACD, BollingerBands,
    rsi, stochastic, macd, bollinger_bands
)

__all__ = [
    # Base classes
    'BaseIndicator', 'MovingAverageBase', 'OscillatorBase',

    # Moving averages
    'SimpleMovingAverage', 'ExponentialMovingAverage', 'WeightedMovingAverage',
    'MovingAverageCrossover', 'sma', 'ema', 'wma', 'ma_crossover',

    # Oscillators
    'RSI', 'Stochastic', 'MACD', 'BollingerBands',
    'rsi', 'stochastic', 'macd', 'bollinger_bands'
]