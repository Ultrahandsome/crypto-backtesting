"""
Base strategy class for all trading strategies.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading strategies."""
    BUY = 1
    SELL = -1
    HOLD = 0


class Position:
    """Represents a trading position."""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float,
                 entry_time: pd.Timestamp, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        """
        Initialize a position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        """
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.is_open = True
    
    def update_pnl(self, current_price: float) -> float:
        """
        Update unrealized PnL.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized PnL
        """
        if self.side == 'long':
            self.pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.size
        
        return self.pnl
    
    def close(self, exit_price: float, exit_time: pd.Timestamp) -> float:
        """
        Close the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            
        Returns:
            Realized PnL
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.is_open = False
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
        
        return self.pnl
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if self.stop_loss is None:
            return False
        
        if self.side == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if self.take_profit is None:
            return False
        
        if self.side == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, initial_capital: float = 100000):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            initial_capital: Initial capital
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Position
        self.closed_positions = []
        self.parameters = {}
        self.signals_history = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate trading signals for a given symbol.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            DataFrame with signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, price: float, 
                              signal_strength: float = 1.0) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal_strength: Signal strength (0-1)
            
        Returns:
            Position size
        """
        pass
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current open positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos.is_open}
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position."""
        if symbol not in self.positions or not self.positions[symbol].is_open:
            return 0.0
        
        position = self.positions[symbol]
        return position.size * current_price
    
    def get_total_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        for symbol, position in self.get_current_positions().items():
            if symbol in current_prices:
                position_value = self.get_position_value(symbol, current_prices[symbol])
                total_value += position_value - (position.size * position.entry_price)
        
        return total_value
    
    def open_position(self, symbol: str, side: str, size: float, price: float,
                     timestamp: pd.Timestamp, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            price: Entry price
            timestamp: Entry timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position opened successfully
        """
        # Check if we already have a position in this symbol
        if symbol in self.positions and self.positions[symbol].is_open:
            logger.warning(f"Already have open position in {symbol}")
            return False
        
        # Check if we have enough capital
        required_capital = size * price
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital for {symbol}: need {required_capital}, have {self.current_capital}")
            return False
        
        # Create position
        position = Position(symbol, side, size, price, timestamp, stop_loss, take_profit)
        self.positions[symbol] = position
        
        # Update capital
        self.current_capital -= required_capital
        
        logger.info(f"Opened {side} position: {symbol} @ {price}, size: {size}")
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: pd.Timestamp,
                      reason: str = "signal") -> Optional[float]:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Realized PnL or None if no position
        """
        if symbol not in self.positions or not self.positions[symbol].is_open:
            logger.warning(f"No open position to close for {symbol}")
            return None
        
        position = self.positions[symbol]
        pnl = position.close(price, timestamp)
        
        # Update capital
        self.current_capital += position.size * price + pnl
        
        # Move to closed positions
        self.closed_positions.append(position)
        
        logger.info(f"Closed {position.side} position: {symbol} @ {price}, PnL: {pnl:.2f}, reason: {reason}")
        return pnl
    
    def check_risk_management(self, current_prices: Dict[str, float], 
                            current_time: pd.Timestamp) -> List[str]:
        """
        Check risk management rules and close positions if needed.
        
        Args:
            current_prices: Current market prices
            current_time: Current timestamp
            
        Returns:
            List of symbols where positions were closed
        """
        closed_symbols = []
        
        for symbol, position in list(self.get_current_positions().items()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check stop loss
            if position.should_stop_loss(current_price):
                self.close_position(symbol, current_price, current_time, "stop_loss")
                closed_symbols.append(symbol)
                continue
            
            # Check take profit
            if position.should_take_profit(current_price):
                self.close_position(symbol, current_price, current_time, "take_profit")
                closed_symbols.append(symbol)
                continue
        
        return closed_symbols
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if not self.closed_positions:
            return {}
        
        pnls = [pos.pnl for pos in self.closed_positions]
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': len([pnl for pnl in pnls if pnl > 0]),
            'losing_trades': len([pnl for pnl in pnls if pnl < 0]),
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'win_rate': len([pnl for pnl in pnls if pnl > 0]) / len(pnls) * 100,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0
        }
