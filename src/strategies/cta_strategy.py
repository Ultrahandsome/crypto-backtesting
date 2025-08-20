"""
CTA (Commodity Trading Advisor) Strategy implementation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from .base_strategy import BaseStrategy, SignalType
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indicators import sma, ema, rsi, ma_crossover

logger = logging.getLogger(__name__)


class CTAStrategy(BaseStrategy):
    """
    CTA Strategy using Moving Average crossovers and RSI filtering.
    
    Entry Rules:
    - Long: Fast MA > Slow MA AND RSI > oversold level
    - Short: Fast MA < Slow MA AND RSI < overbought level
    
    Exit Rules:
    - Opposite signal
    - Stop loss hit
    - Take profit hit
    """
    
    def __init__(self, 
                 fast_ma_period: int = 10,
                 slow_ma_period: int = 30,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 ma_type: str = 'sma',
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 position_size_pct: float = 0.1,
                 initial_capital: float = 100000):
        """
        Initialize CTA Strategy.
        
        Args:
            fast_ma_period: Fast moving average period
            slow_ma_period: Slow moving average period
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            ma_type: Moving average type ('sma', 'ema')
            stop_loss_pct: Stop loss percentage (0.02 = 2%)
            take_profit_pct: Take profit percentage (0.06 = 6%)
            position_size_pct: Position size as percentage of capital
            initial_capital: Initial capital
        """
        super().__init__("CTA_Strategy", initial_capital)
        
        # Strategy parameters
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ma_type = ma_type.lower()
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct
        
        # Store parameters
        self.parameters = {
            'fast_ma_period': fast_ma_period,
            'slow_ma_period': slow_ma_period,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'ma_type': ma_type,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size_pct': position_size_pct
        }
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate strategy parameters."""
        if self.fast_ma_period >= self.slow_ma_period:
            raise ValueError("Fast MA period must be less than slow MA period")
        
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ValueError("RSI thresholds must be: 0 < oversold < overbought < 100")
        
        if self.ma_type not in ['sma', 'ema']:
            raise ValueError("MA type must be 'sma' or 'ema'")
        
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("Position size percentage must be between 0 and 1")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate CTA trading signals.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            DataFrame with signals and indicators
        """
        if len(data) < max(self.slow_ma_period, self.rsi_period):
            logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
            return pd.DataFrame(index=data.index)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate indicators
        if self.ma_type == 'sma':
            signals['fast_ma'] = sma(data, period=self.fast_ma_period)
            signals['slow_ma'] = sma(data, period=self.slow_ma_period)
        else:  # ema
            signals['fast_ma'] = ema(data, period=self.fast_ma_period)
            signals['slow_ma'] = ema(data, period=self.slow_ma_period)
        
        signals['rsi'] = rsi(data, period=self.rsi_period)
        signals['close'] = data['close']
        
        # Generate base signals
        signals['ma_signal'] = 0
        signals.loc[signals['fast_ma'] > signals['slow_ma'], 'ma_signal'] = 1  # Bullish
        signals.loc[signals['fast_ma'] < signals['slow_ma'], 'ma_signal'] = -1  # Bearish
        
        # RSI filter
        signals['rsi_filter'] = 0
        signals.loc[signals['rsi'] > self.rsi_oversold, 'rsi_filter'] = 1  # Allow long
        signals.loc[signals['rsi'] < self.rsi_overbought, 'rsi_filter'] = -1  # Allow short
        
        # Combined signals
        signals['signal'] = 0
        
        # Long signal: MA bullish AND RSI not oversold
        long_condition = (signals['ma_signal'] == 1) & (signals['rsi'] > self.rsi_oversold)
        signals.loc[long_condition, 'signal'] = 1
        
        # Short signal: MA bearish AND RSI not overbought  
        short_condition = (signals['ma_signal'] == -1) & (signals['rsi'] < self.rsi_overbought)
        signals.loc[short_condition, 'signal'] = -1
        
        # Detect signal changes
        signals['signal_change'] = signals['signal'].diff()
        signals['entry_long'] = signals['signal_change'] == 2  # 0 or -1 to 1
        signals['entry_short'] = signals['signal_change'] == -2  # 0 or 1 to -1
        signals['exit_signal'] = (signals['signal_change'] != 0) & (signals['signal'] == 0)
        
        # Calculate signal strength (for position sizing)
        signals['signal_strength'] = 0.5  # Base strength
        
        # Increase strength based on RSI extremes
        rsi_strength = np.where(
            signals['rsi'] < self.rsi_oversold, 
            (self.rsi_oversold - signals['rsi']) / self.rsi_oversold,
            np.where(
                signals['rsi'] > self.rsi_overbought,
                (signals['rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought),
                0
            )
        )
        signals['signal_strength'] += rsi_strength * 0.3
        signals['signal_strength'] = np.clip(signals['signal_strength'], 0.1, 1.0)
        
        logger.debug(f"Generated signals for {symbol}: "
                    f"{signals['entry_long'].sum()} long entries, "
                    f"{signals['entry_short'].sum()} short entries")
        
        return signals
    
    def calculate_position_size(self, symbol: str, price: float, 
                              signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on available capital and signal strength.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal_strength: Signal strength (0-1)
            
        Returns:
            Position size (number of shares/units)
        """
        # Base position value as percentage of current capital
        base_position_value = self.current_capital * self.position_size_pct
        
        # Adjust by signal strength
        adjusted_position_value = base_position_value * signal_strength
        
        # Calculate number of shares/units
        position_size = adjusted_position_value / price
        
        # Ensure we don't exceed available capital
        max_position_value = self.current_capital * 0.95  # Leave 5% buffer
        max_position_size = max_position_value / price
        
        position_size = min(position_size, max_position_size)
        
        logger.debug(f"Position size for {symbol}: {position_size:.4f} units "
                    f"(value: ${position_size * price:.2f})")
        
        return position_size
    
    def calculate_stop_loss_take_profit(self, entry_price: float, side: str) -> tuple:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        if side == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        return stop_loss, take_profit
    
    def process_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Process signals and generate trade actions.
        
        Args:
            data: OHLCV DataFrame with signals
            symbol: Trading symbol
            
        Returns:
            List of trade actions
        """
        actions = []
        
        for timestamp, row in data.iterrows():
            current_price = row['close']
            
            # Check risk management first
            closed_symbols = self.check_risk_management({symbol: current_price}, timestamp)
            
            # Process entry signals
            if row.get('entry_long', False):
                if symbol not in self.get_current_positions():
                    size = self.calculate_position_size(symbol, current_price, row.get('signal_strength', 1.0))
                    if size > 0:
                        stop_loss, take_profit = self.calculate_stop_loss_take_profit(current_price, 'long')
                        
                        if self.open_position(symbol, 'long', size, current_price, timestamp, 
                                            stop_loss, take_profit):
                            actions.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'action': 'buy',
                                'price': current_price,
                                'size': size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            })
            
            elif row.get('entry_short', False):
                if symbol not in self.get_current_positions():
                    size = self.calculate_position_size(symbol, current_price, row.get('signal_strength', 1.0))
                    if size > 0:
                        stop_loss, take_profit = self.calculate_stop_loss_take_profit(current_price, 'short')
                        
                        if self.open_position(symbol, 'short', size, current_price, timestamp,
                                            stop_loss, take_profit):
                            actions.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'action': 'sell',
                                'price': current_price,
                                'size': size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            })
            
            # Process exit signals
            elif row.get('exit_signal', False):
                if symbol in self.get_current_positions():
                    pnl = self.close_position(symbol, current_price, timestamp, "signal")
                    if pnl is not None:
                        actions.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'close',
                            'price': current_price,
                            'pnl': pnl
                        })
        
        return actions
