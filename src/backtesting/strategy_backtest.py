"""
Strategy-specific backtesting implementation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from .base_backtest import BacktestEngine, OrderType
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Backtester specifically designed for strategy testing.
    """
    
    def __init__(self,
                 strategy: BaseStrategy,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize strategy backtester.
        
        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            commission_rate: Commission rate
            slippage_rate: Slippage rate
            benchmark_symbol: Benchmark symbol for comparison
        """
        self.strategy = strategy
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        self.benchmark_symbol = benchmark_symbol
        
        # Backtest results
        self.results = {}
        self.benchmark_data = None
        
    def run_backtest(self,
                    data: Dict[str, pd.DataFrame],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    warmup_period: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            data: Market data {symbol: OHLCV DataFrame}
            start_date: Start date for backtest
            end_date: End date for backtest
            warmup_period: Warmup period for indicators
            
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting strategy backtest...")
        
        # Reset engine and strategy
        self.engine.reset()
        self.strategy.current_capital = self.engine.initial_capital
        self.strategy.positions = {}
        self.strategy.closed_positions = []
        
        # Prepare data
        aligned_data = self._align_data(data, start_date, end_date)
        if not aligned_data:
            raise ValueError("No valid data for backtesting")
        
        # Get date range
        all_dates = sorted(set().union(*[df.index for df in aligned_data.values()]))
        
        logger.info(f"Backtesting from {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Assets: {list(aligned_data.keys())}")
        
        # Run backtest day by day
        for i, current_date in enumerate(all_dates):
            if i < warmup_period:
                continue  # Skip warmup period
                
            # Get current market data
            current_data = {}
            current_prices = {}
            
            for symbol, df in aligned_data.items():
                if current_date in df.index:
                    current_data[symbol] = df.loc[current_date]
                    current_prices[symbol] = df.loc[current_date]['close']
            
            if not current_data:
                continue
            
            # Process pending orders first
            self.engine.process_orders(current_data, current_date)
            
            # Generate signals for each symbol
            for symbol, df in aligned_data.items():
                if current_date not in df.index:
                    continue
                    
                # Get historical data up to current date
                historical_data = df.loc[:current_date].copy()
                if len(historical_data) < warmup_period:
                    continue
                
                # Generate signals
                signals = self.strategy.generate_signals(historical_data, symbol)
                if signals.empty or current_date not in signals.index:
                    continue
                
                current_signal = signals.loc[current_date]
                
                # Process entry signals
                if current_signal.get('entry_long', False):
                    self._process_entry_signal(symbol, 'buy', current_data[symbol], 
                                             current_signal, current_date)
                elif current_signal.get('entry_short', False):
                    self._process_entry_signal(symbol, 'sell', current_data[symbol], 
                                             current_signal, current_date)
                
                # Process exit signals
                elif current_signal.get('exit_signal', False):
                    self._process_exit_signal(symbol, current_data[symbol], current_date)
            
            # Update portfolio value
            self.engine.update_portfolio_value(current_prices, current_date)
            
            # Log progress periodically
            if i % 100 == 0:
                logger.debug(f"Processed {i}/{len(all_dates)} days, "
                           f"Portfolio value: ${self.engine.portfolio_value:.2f}")
        
        # Generate results
        self.results = self._generate_results(aligned_data)
        
        logger.info(f"Backtest completed. Final portfolio value: "
                   f"${self.engine.portfolio_value:.2f}")
        
        return self.results
    
    def _align_data(self, data: Dict[str, pd.DataFrame], 
                   start_date: Optional[str], 
                   end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """Align data across all symbols."""
        aligned_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Filter by date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            if not df.empty:
                aligned_data[symbol] = df
        
        return aligned_data
    
    def _process_entry_signal(self, symbol: str, side: str, market_data: pd.Series,
                            signal_data: pd.Series, timestamp: pd.Timestamp):
        """Process entry signal."""
        # Check if we already have a position
        if symbol in self.engine.positions and self.engine.positions[symbol] != 0:
            return
        
        # Calculate position size
        current_price = market_data['close']
        signal_strength = signal_data.get('signal_strength', 1.0)
        
        # Use strategy's position sizing
        position_size = self.strategy.calculate_position_size(
            symbol, current_price, signal_strength
        )
        
        if position_size <= 0:
            return
        
        # Place market order
        order_id = self.engine.place_order(
            symbol=symbol,
            side=side,
            quantity=position_size,
            order_type=OrderType.MARKET,
            timestamp=timestamp
        )
        
        logger.debug(f"Entry signal: {side} {position_size:.4f} {symbol} @ {current_price:.4f}")
    
    def _process_exit_signal(self, symbol: str, market_data: pd.Series, timestamp: pd.Timestamp):
        """Process exit signal."""
        if symbol not in self.engine.positions or self.engine.positions[symbol] == 0:
            return
        
        position_size = abs(self.engine.positions[symbol])
        side = 'sell' if self.engine.positions[symbol] > 0 else 'buy'
        
        # Place market order to close position
        order_id = self.engine.place_order(
            symbol=symbol,
            side=side,
            quantity=position_size,
            order_type=OrderType.MARKET,
            timestamp=timestamp
        )
        
        current_price = market_data['close']
        logger.debug(f"Exit signal: {side} {position_size:.4f} {symbol} @ {current_price:.4f}")
    
    def _generate_results(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        results = {
            'summary': self._calculate_summary_stats(),
            'trades': [self._trade_to_dict(trade) for trade in self.engine.trades],
            'equity_curve': pd.DataFrame(self.engine.equity_curve),
            'positions': self.engine.positions.copy(),
            'orders': [self._order_to_dict(order) for order in self.engine.orders],
            'performance_metrics': self._calculate_performance_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'benchmark_comparison': self._calculate_benchmark_comparison(data)
        }
        
        return results
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.engine.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.engine.equity_curve)
        
        return {
            'initial_capital': self.engine.initial_capital,
            'final_portfolio_value': self.engine.portfolio_value,
            'total_return': (self.engine.portfolio_value - self.engine.initial_capital) / self.engine.initial_capital,
            'total_trades': len(self.engine.trades),
            'winning_trades': len([t for t in self.engine.trades if t.pnl > 0]),
            'losing_trades': len([t for t in self.engine.trades if t.pnl < 0]),
            'win_rate': len([t for t in self.engine.trades if t.pnl > 0]) / max(len(self.engine.trades), 1),
            'total_commission': sum(t.commission for t in self.engine.trades),
            'total_slippage': sum(t.slippage for t in self.engine.trades),
            'start_date': equity_df['timestamp'].iloc[0] if not equity_df.empty else None,
            'end_date': equity_df['timestamp'].iloc[-1] if not equity_df.empty else None,
            'duration_days': (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days if len(equity_df) > 1 else 0
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.engine.returns:
            return {}
        
        returns = pd.Series(self.engine.returns)
        
        # Basic metrics
        total_return = (self.engine.portfolio_value - self.engine.initial_capital) / self.engine.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        equity_curve = pd.Series([eq['portfolio_value'] for eq in self.engine.equity_curve])
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if self.engine.trades:
            trade_pnls = [t.pnl for t in self.engine.trades]
            avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
            avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if any(pnl < 0 for pnl in trade_pnls) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.engine.returns:
            return {}
        
        returns = pd.Series(self.engine.returns)
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        # Sortino ratio
        sortino_ratio = (returns.mean() * 252 - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'skewness': returns.skew() if len(returns) > 2 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 3 else 0
        }
    
    def _calculate_benchmark_comparison(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        if self.benchmark_symbol not in data or not self.engine.equity_curve:
            return {}
        
        benchmark_data = data[self.benchmark_symbol]
        equity_df = pd.DataFrame(self.engine.equity_curve)
        
        # Align benchmark with equity curve dates
        start_date = equity_df['timestamp'].iloc[0]
        end_date = equity_df['timestamp'].iloc[-1]
        
        benchmark_aligned = benchmark_data[(benchmark_data.index >= start_date) & 
                                         (benchmark_data.index <= end_date)]
        
        if benchmark_aligned.empty:
            return {}
        
        # Calculate benchmark return
        benchmark_return = (benchmark_aligned['close'].iloc[-1] - benchmark_aligned['close'].iloc[0]) / benchmark_aligned['close'].iloc[0]
        strategy_return = (self.engine.portfolio_value - self.engine.initial_capital) / self.engine.initial_capital
        
        return {
            'benchmark_return': benchmark_return,
            'excess_return': strategy_return - benchmark_return,
            'tracking_error': 0.0,  # Would need daily alignment for proper calculation
            'information_ratio': 0.0  # Would need daily alignment for proper calculation
        }
    
    def _trade_to_dict(self, trade: Any) -> Dict[str, Any]:
        """Convert trade object to dictionary."""
        return {
            'symbol': trade.symbol,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'duration_days': trade.duration.days,
            'entry_reason': trade.entry_reason,
            'exit_reason': trade.exit_reason,
            'mfe': trade.max_favorable_excursion,
            'mae': trade.max_adverse_excursion
        }
    
    def _order_to_dict(self, order: Any) -> Dict[str, Any]:
        """Convert order object to dictionary."""
        return {
            'id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'type': order.order_type.value,
            'price': order.price,
            'stop_price': order.stop_price,
            'timestamp': order.timestamp,
            'status': order.status.value,
            'filled_price': order.filled_price,
            'filled_quantity': order.filled_quantity,
            'commission': order.commission,
            'slippage': order.slippage
        }
