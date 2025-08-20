"""
Base backtesting framework for trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    slippage: float
    duration: timedelta
    entry_reason: str
    exit_reason: str
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


class BacktestEngine:
    """
    Comprehensive backtesting engine with advanced features.
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 margin_rate: float = 1.0,
                 interest_rate: float = 0.02):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_rate: Slippage rate (0.0005 = 0.05%)
            margin_rate: Margin requirement (1.0 = 100%, 0.5 = 50%)
            interest_rate: Annual interest rate for margin
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.margin_rate = margin_rate
        self.interest_rate = interest_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.portfolio_value = initial_capital
        
        # Order management
        self.orders = []
        self.order_counter = 0
        
        # Trade tracking
        self.trades = []
        self.open_positions = {}  # symbol -> position info
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.returns = []
        
        # Logging
        self.trade_log = []
        self.order_log = []
        
    def reset(self):
        """Reset the backtesting engine to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.orders = []
        self.order_counter = 0
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        self.drawdown_curve = []
        self.returns = []
        self.trade_log = []
        self.order_log = []
        
    def place_order(self,
                   symbol: str,
                   side: str,
                   quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   timestamp: Optional[pd.Timestamp] = None) -> str:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            timestamp: Order timestamp
            
        Returns:
            Order ID
        """
        order_id = f"ORDER_{self.order_counter:06d}"
        self.order_counter += 1
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            timestamp=timestamp
        )
        
        self.orders.append(order)
        self.order_log.append({
            'timestamp': timestamp,
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'type': order_type.value,
            'price': price,
            'status': 'placed'
        })
        
        logger.debug(f"Order placed: {order_id} - {side} {quantity} {symbol}")
        return order_id
    
    def process_orders(self, current_data: Dict[str, pd.Series], timestamp: pd.Timestamp):
        """
        Process pending orders against current market data.
        
        Args:
            current_data: Current market data {symbol: OHLCV series}
            timestamp: Current timestamp
        """
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
                
            if order.symbol not in current_data:
                continue
                
            market_data = current_data[order.symbol]
            filled = False
            fill_price = None
            
            # Determine fill price based on order type
            if order.order_type == OrderType.MARKET:
                # Market orders fill at open price (assuming order placed at previous close)
                fill_price = market_data['open']
                filled = True
                
            elif order.order_type == OrderType.LIMIT:
                if order.side == 'buy' and market_data['low'] <= order.price:
                    fill_price = min(order.price, market_data['open'])
                    filled = True
                elif order.side == 'sell' and market_data['high'] >= order.price:
                    fill_price = max(order.price, market_data['open'])
                    filled = True
                    
            elif order.order_type == OrderType.STOP:
                if order.side == 'buy' and market_data['high'] >= order.stop_price:
                    fill_price = max(order.stop_price, market_data['open'])
                    filled = True
                elif order.side == 'sell' and market_data['low'] <= order.stop_price:
                    fill_price = min(order.stop_price, market_data['open'])
                    filled = True
            
            if filled:
                self._fill_order(order, fill_price, timestamp)
    
    def _fill_order(self, order: Order, fill_price: float, timestamp: pd.Timestamp):
        """Fill an order at the specified price."""
        # Calculate slippage
        slippage_amount = fill_price * self.slippage_rate
        if order.side == 'buy':
            fill_price += slippage_amount
        else:
            fill_price -= slippage_amount
        
        # Calculate commission
        commission = abs(order.quantity * fill_price * self.commission_rate)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage_amount * order.quantity
        
        # Update portfolio
        self._update_portfolio(order)
        
        # Log the fill
        self.order_log.append({
            'timestamp': timestamp,
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'fill_price': fill_price,
            'commission': commission,
            'slippage': order.slippage,
            'status': 'filled'
        })
        
        logger.debug(f"Order filled: {order.id} - {order.side} {order.quantity} "
                    f"{order.symbol} @ {fill_price:.4f}")
    
    def _update_portfolio(self, order: Order):
        """Update portfolio state after order fill."""
        symbol = order.symbol
        quantity = order.filled_quantity if order.side == 'buy' else -order.filled_quantity
        cost = order.filled_quantity * order.filled_price + order.commission
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
        
        # Update cash
        if order.side == 'buy':
            self.cash -= cost
        else:
            self.cash += cost - order.commission  # Commission already included in cost
        
        # Track position for trade analysis
        self._track_position_change(order)
    
    def _track_position_change(self, order: Order):
        """Track position changes for trade analysis."""
        symbol = order.symbol
        
        if symbol not in self.open_positions:
            # Opening new position
            if order.side == 'buy':
                self.open_positions[symbol] = {
                    'side': 'long',
                    'quantity': order.filled_quantity,
                    'entry_price': order.filled_price,
                    'entry_time': order.timestamp,
                    'entry_reason': 'signal',
                    'max_price': order.filled_price,
                    'min_price': order.filled_price
                }
            else:
                self.open_positions[symbol] = {
                    'side': 'short',
                    'quantity': order.filled_quantity,
                    'entry_price': order.filled_price,
                    'entry_time': order.timestamp,
                    'entry_reason': 'signal',
                    'max_price': order.filled_price,
                    'min_price': order.filled_price
                }
        else:
            # Closing or modifying existing position
            pos = self.open_positions[symbol]
            
            if ((pos['side'] == 'long' and order.side == 'sell') or
                (pos['side'] == 'short' and order.side == 'buy')):
                
                # Closing position - create trade record
                trade = Trade(
                    symbol=symbol,
                    entry_time=pos['entry_time'],
                    exit_time=order.timestamp,
                    side=pos['side'],
                    entry_price=pos['entry_price'],
                    exit_price=order.filled_price,
                    quantity=min(pos['quantity'], order.filled_quantity),
                    pnl=self._calculate_trade_pnl(pos, order.filled_price, order.filled_quantity),
                    commission=order.commission,
                    slippage=order.slippage,
                    duration=order.timestamp - pos['entry_time'],
                    entry_reason=pos['entry_reason'],
                    exit_reason='signal',
                    max_favorable_excursion=self._calculate_mfe(pos),
                    max_adverse_excursion=self._calculate_mae(pos)
                )
                
                self.trades.append(trade)
                
                # Update or remove position
                if pos['quantity'] <= order.filled_quantity:
                    del self.open_positions[symbol]
                else:
                    pos['quantity'] -= order.filled_quantity
    
    def _calculate_trade_pnl(self, position: Dict, exit_price: float, exit_quantity: float) -> float:
        """Calculate PnL for a trade."""
        if position['side'] == 'long':
            return (exit_price - position['entry_price']) * exit_quantity
        else:
            return (position['entry_price'] - exit_price) * exit_quantity
    
    def _calculate_mfe(self, position: Dict) -> float:
        """Calculate Maximum Favorable Excursion."""
        if position['side'] == 'long':
            return position['max_price'] - position['entry_price']
        else:
            return position['entry_price'] - position['min_price']
    
    def _calculate_mae(self, position: Dict) -> float:
        """Calculate Maximum Adverse Excursion."""
        if position['side'] == 'long':
            return position['entry_price'] - position['min_price']
        else:
            return position['max_price'] - position['entry_price']
    
    def update_portfolio_value(self, current_prices: Dict[str, float], timestamp: pd.Timestamp):
        """
        Update portfolio value based on current prices.
        
        Args:
            current_prices: Current prices {symbol: price}
            timestamp: Current timestamp
        """
        portfolio_value = self.cash
        
        # Add value of positions
        for symbol, quantity in self.positions.items():
            if symbol in current_prices and quantity != 0:
                portfolio_value += quantity * current_prices[symbol]
        
        self.portfolio_value = portfolio_value
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash
        })
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.returns.append(daily_return)
        
        # Update open positions with current prices for MFE/MAE tracking
        for symbol, pos in self.open_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                pos['max_price'] = max(pos['max_price'], current_price)
                pos['min_price'] = min(pos['min_price'], current_price)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'open_positions': len(self.open_positions),
            'total_trades': len(self.trades),
            'pending_orders': len([o for o in self.orders if o.status == OrderStatus.PENDING])
        }
