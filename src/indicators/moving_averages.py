"""
Moving Average indicators implementation.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
import logging
from .base_indicator import MovingAverageBase

logger = logging.getLogger(__name__)


class SimpleMovingAverage(MovingAverageBase):
    """Simple Moving Average (SMA) indicator."""
    
    def __init__(self, period: int = 20):
        """
        Initialize SMA indicator.
        
        Args:
            period: Period for SMA calculation
        """
        super().__init__("SMA", period)
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate SMA on (default: 'close')
            
        Returns:
            Series with SMA values
        """
        if not self.validate_data(data, [column]):
            return pd.Series(dtype=float)
        
        if not self.validate_period(len(data)):
            return pd.Series(dtype=float)
        
        sma = data[column].rolling(window=self.period, min_periods=1).mean()
        logger.debug(f"Calculated SMA({self.period}) for {len(sma)} periods")
        
        return sma


class ExponentialMovingAverage(MovingAverageBase):
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int = 20, alpha: Optional[float] = None):
        """
        Initialize EMA indicator.
        
        Args:
            period: Period for EMA calculation
            alpha: Smoothing factor (if None, calculated as 2/(period+1))
        """
        super().__init__("EMA", period)
        self.alpha = alpha if alpha is not None else 2.0 / (period + 1)
        self._parameters['alpha'] = self.alpha
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate EMA on (default: 'close')
            
        Returns:
            Series with EMA values
        """
        if not self.validate_data(data, [column]):
            return pd.Series(dtype=float)
        
        if not self.validate_period(len(data)):
            return pd.Series(dtype=float)
        
        ema = data[column].ewm(alpha=self.alpha, adjust=False).mean()
        logger.debug(f"Calculated EMA({self.period}) for {len(ema)} periods")
        
        return ema


class WeightedMovingAverage(MovingAverageBase):
    """Weighted Moving Average (WMA) indicator."""
    
    def __init__(self, period: int = 20):
        """
        Initialize WMA indicator.
        
        Args:
            period: Period for WMA calculation
        """
        super().__init__("WMA", period)
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Weighted Moving Average.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate WMA on (default: 'close')
            
        Returns:
            Series with WMA values
        """
        if not self.validate_data(data, [column]):
            return pd.Series(dtype=float)
        
        if not self.validate_period(len(data)):
            return pd.Series(dtype=float)
        
        weights = np.arange(1, self.period + 1)
        
        def wma_calc(x):
            if len(x) < self.period:
                # For partial periods, use available data
                w = weights[:len(x)]
                return np.average(x, weights=w)
            return np.average(x, weights=weights)
        
        wma = data[column].rolling(window=self.period, min_periods=1).apply(wma_calc, raw=True)
        logger.debug(f"Calculated WMA({self.period}) for {len(wma)} periods")
        
        return wma


class MovingAverageCrossover:
    """Moving Average Crossover signals."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, 
                 ma_type: str = 'sma'):
        """
        Initialize MA Crossover.
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.lower()
        
        # Initialize MA indicators
        if self.ma_type == 'sma':
            self.fast_ma = SimpleMovingAverage(fast_period)
            self.slow_ma = SimpleMovingAverage(slow_period)
        elif self.ma_type == 'ema':
            self.fast_ma = ExponentialMovingAverage(fast_period)
            self.slow_ma = ExponentialMovingAverage(slow_period)
        elif self.ma_type == 'wma':
            self.fast_ma = WeightedMovingAverage(fast_period)
            self.slow_ma = WeightedMovingAverage(slow_period)
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate MA crossover signals.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate on
            
        Returns:
            DataFrame with MA values and signals
        """
        result = pd.DataFrame(index=data.index)
        
        # Calculate moving averages
        result['fast_ma'] = self.fast_ma.calculate(data, column)
        result['slow_ma'] = self.slow_ma.calculate(data, column)
        
        # Calculate crossover signals
        result['signal'] = 0
        result.loc[result['fast_ma'] > result['slow_ma'], 'signal'] = 1  # Bullish
        result.loc[result['fast_ma'] < result['slow_ma'], 'signal'] = -1  # Bearish
        
        # Detect crossover points
        result['crossover'] = result['signal'].diff()
        result['bullish_crossover'] = result['crossover'] == 2  # -1 to 1
        result['bearish_crossover'] = result['crossover'] == -2  # 1 to -1
        
        logger.debug(f"Calculated MA crossover signals: "
                    f"{result['bullish_crossover'].sum()} bullish, "
                    f"{result['bearish_crossover'].sum()} bearish")
        
        return result


# Convenience functions
def sma(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """Calculate Simple Moving Average."""
    indicator = SimpleMovingAverage(period)
    return indicator.calculate(data, column)


def ema(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """Calculate Exponential Moving Average."""
    indicator = ExponentialMovingAverage(period)
    return indicator.calculate(data, column)


def wma(data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """Calculate Weighted Moving Average."""
    indicator = WeightedMovingAverage(period)
    return indicator.calculate(data, column)


def ma_crossover(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 30,
                ma_type: str = 'sma', column: str = 'close') -> pd.DataFrame:
    """Calculate Moving Average crossover signals."""
    crossover = MovingAverageCrossover(fast_period, slow_period, ma_type)
    return crossover.calculate(data, column)
