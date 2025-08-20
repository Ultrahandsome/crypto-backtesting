"""
Base class for technical indicators.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(self, name: str):
        """
        Initialize the base indicator.
        
        Args:
            name: Name of the indicator
        """
        self.name = name
        self._parameters = {}
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the indicator values.
        
        Args:
            data: OHLCV DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Series or DataFrame with indicator values
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input DataFrame
            required_columns: List of required columns
            
        Returns:
            True if data is valid
        """
        if data.empty:
            logger.error(f"{self.name}: Input data is empty")
            return False
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"{self.name}: Missing columns: {missing_columns}")
                return False
        
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return self._parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """Set indicator parameters."""
        self._parameters.update(kwargs)


class MovingAverageBase(BaseIndicator):
    """Base class for moving average indicators."""
    
    def __init__(self, name: str, period: int = 20):
        """
        Initialize moving average base.
        
        Args:
            name: Name of the indicator
            period: Period for calculation
        """
        super().__init__(name)
        self.period = period
        self._parameters['period'] = period
    
    def validate_period(self, data_length: int) -> bool:
        """
        Validate that period is appropriate for data length.
        
        Args:
            data_length: Length of input data
            
        Returns:
            True if period is valid
        """
        if self.period <= 0:
            logger.error(f"{self.name}: Period must be positive, got {self.period}")
            return False
        
        if self.period > data_length:
            logger.warning(f"{self.name}: Period ({self.period}) > data length ({data_length})")
            return False
        
        return True


class OscillatorBase(BaseIndicator):
    """Base class for oscillator indicators."""
    
    def __init__(self, name: str, period: int = 14, 
                 overbought: float = 70, oversold: float = 30):
        """
        Initialize oscillator base.
        
        Args:
            name: Name of the indicator
            period: Period for calculation
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__(name)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self._parameters.update({
            'period': period,
            'overbought': overbought,
            'oversold': oversold
        })
    
    def get_signals(self, values: pd.Series) -> pd.DataFrame:
        """
        Generate buy/sell signals based on oscillator levels.
        
        Args:
            values: Oscillator values
            
        Returns:
            DataFrame with signal columns
        """
        signals = pd.DataFrame(index=values.index)
        signals['value'] = values
        signals['overbought'] = values > self.overbought
        signals['oversold'] = values < self.oversold
        signals['buy_signal'] = (values < self.oversold) & (values.shift(1) >= self.oversold)
        signals['sell_signal'] = (values > self.overbought) & (values.shift(1) <= self.overbought)
        
        return signals


def safe_divide(numerator: Union[pd.Series, float], 
                denominator: Union[pd.Series, float], 
                fill_value: float = 0.0) -> Union[pd.Series, float]:
    """
    Safely divide two values, handling division by zero.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero
        
    Returns:
        Division result with safe handling of zero division
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value)
    else:
        return numerator / denominator if denominator != 0 else fill_value


def rolling_window_safe(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Apply rolling window with safe handling of edge cases.
    
    Args:
        series: Input series
        window: Window size
        min_periods: Minimum periods required
        
    Returns:
        Series with rolling window applied
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    return series.rolling(window=window, min_periods=min_periods)
