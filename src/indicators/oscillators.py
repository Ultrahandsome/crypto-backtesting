"""
Oscillator indicators implementation.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import logging
from .base_indicator import OscillatorBase, safe_divide

logger = logging.getLogger(__name__)


class RSI(OscillatorBase):
    """Relative Strength Index (RSI) indicator."""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator.
        
        Args:
            period: Period for RSI calculation
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
        """
        super().__init__("RSI", period, overbought, oversold)
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate RSI on (default: 'close')
            
        Returns:
            Series with RSI values (0-100)
        """
        if not self.validate_data(data, [column]):
            return pd.Series(dtype=float)
        
        if len(data) < self.period:
            logger.warning(f"RSI: Insufficient data ({len(data)} < {self.period})")
            return pd.Series(dtype=float, index=data.index)
        
        # Calculate price changes
        delta = data[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using EMA
        avg_gains = gains.ewm(alpha=1/self.period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = safe_divide(avg_gains, avg_losses, fill_value=0)
        rsi = 100 - (100 / (1 + rs))
        
        logger.debug(f"Calculated RSI({self.period}) for {len(rsi)} periods")
        
        return rsi
    
    def get_signals(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Get RSI signals with buy/sell recommendations.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate RSI on
            
        Returns:
            DataFrame with RSI values and signals
        """
        rsi_values = self.calculate(data, column)
        signals = super().get_signals(rsi_values)
        
        return signals


class Stochastic(OscillatorBase):
    """Stochastic Oscillator indicator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, 
                 overbought: float = 80, oversold: float = 20):
        """
        Initialize Stochastic indicator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D (SMA of %K)
            overbought: Overbought threshold (default: 80)
            oversold: Oversold threshold (default: 20)
        """
        super().__init__("Stochastic", k_period, overbought, oversold)
        self.k_period = k_period
        self.d_period = d_period
        self._parameters['k_period'] = k_period
        self._parameters['d_period'] = d_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with %K and %D values
        """
        required_columns = ['high', 'low', 'close']
        if not self.validate_data(data, required_columns):
            return pd.DataFrame()
        
        if len(data) < self.k_period:
            logger.warning(f"Stochastic: Insufficient data ({len(data)} < {self.k_period})")
            return pd.DataFrame(index=data.index)
        
        # Calculate %K
        lowest_low = data['low'].rolling(window=self.k_period, min_periods=1).min()
        highest_high = data['high'].rolling(window=self.k_period, min_periods=1).max()
        
        k_percent = 100 * safe_divide(
            data['close'] - lowest_low,
            highest_high - lowest_low,
            fill_value=50
        )
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=self.d_period, min_periods=1).mean()
        
        result = pd.DataFrame(index=data.index)
        result['%K'] = k_percent
        result['%D'] = d_percent
        
        logger.debug(f"Calculated Stochastic({self.k_period}, {self.d_period}) for {len(result)} periods")
        
        return result


class MACD:
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = "MACD"
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate MACD on (default: 'close')
            
        Returns:
            DataFrame with MACD line, Signal line, and Histogram
        """
        if column not in data.columns:
            logger.error(f"MACD: Column '{column}' not found in data")
            return pd.DataFrame()
        
        if len(data) < self.slow_period:
            logger.warning(f"MACD: Insufficient data ({len(data)} < {self.slow_period})")
            return pd.DataFrame(index=data.index)
        
        # Calculate EMAs
        fast_ema = data[column].ewm(span=self.fast_period).mean()
        slow_ema = data[column].ewm(span=self.slow_period).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        result = pd.DataFrame(index=data.index)
        result['MACD'] = macd_line
        result['Signal'] = signal_line
        result['Histogram'] = histogram
        
        # Generate signals
        result['bullish_crossover'] = (
            (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        )
        result['bearish_crossover'] = (
            (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        )
        
        logger.debug(f"Calculated MACD({self.fast_period}, {self.slow_period}, {self.signal_period}) "
                    f"for {len(result)} periods")
        
        return result


class BollingerBands:
    """Bollinger Bands indicator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Period for moving average and standard deviation
            std_dev: Number of standard deviations for bands
        """
        self.period = period
        self.std_dev = std_dev
        self.name = "BollingerBands"
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: OHLCV DataFrame
            column: Column to calculate bands on (default: 'close')
            
        Returns:
            DataFrame with Upper Band, Middle Band (SMA), and Lower Band
        """
        if column not in data.columns:
            logger.error(f"BollingerBands: Column '{column}' not found in data")
            return pd.DataFrame()
        
        if len(data) < self.period:
            logger.warning(f"BollingerBands: Insufficient data ({len(data)} < {self.period})")
            return pd.DataFrame(index=data.index)
        
        # Calculate middle band (SMA)
        middle_band = data[column].rolling(window=self.period, min_periods=1).mean()
        
        # Calculate standard deviation
        std = data[column].rolling(window=self.period, min_periods=1).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        result = pd.DataFrame(index=data.index)
        result['Upper'] = upper_band
        result['Middle'] = middle_band
        result['Lower'] = lower_band
        result['Width'] = upper_band - lower_band
        result['%B'] = safe_divide(data[column] - lower_band, upper_band - lower_band, 0.5)
        
        logger.debug(f"Calculated Bollinger Bands({self.period}, {self.std_dev}) "
                    f"for {len(result)} periods")
        
        return result


# Convenience functions
def rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """Calculate RSI."""
    indicator = RSI(period)
    return indicator.calculate(data, column)


def stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    indicator = Stochastic(k_period, d_period)
    return indicator.calculate(data)


def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
         signal: int = 9, column: str = 'close') -> pd.DataFrame:
    """Calculate MACD."""
    indicator = MACD(fast, slow, signal)
    return indicator.calculate(data, column)


def bollinger_bands(data: pd.DataFrame, period: int = 20, 
                   std_dev: float = 2.0, column: str = 'close') -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    indicator = BollingerBands(period, std_dev)
    return indicator.calculate(data, column)
