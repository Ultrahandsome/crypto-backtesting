"""
Comprehensive charting and visualization tools for trading strategies.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StrategyVisualizer:
    """
    Comprehensive visualization tools for trading strategies.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize strategy visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_strategy_overview(self,
                             data: pd.DataFrame,
                             signals: pd.DataFrame,
                             trades: Optional[List[Dict]] = None,
                             title: str = "Strategy Overview",
                             save_path: Optional[str] = None) -> None:
        """
        Create comprehensive strategy overview chart.
        
        Args:
            data: OHLCV data
            signals: Strategy signals
            trades: Trade data
            title: Chart title
            save_path: Path to save chart
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Moving Averages with Entry/Exit Points
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=2, alpha=0.8)
        
        if 'fast_ma' in signals.columns:
            ax1.plot(signals.index, signals['fast_ma'], label='Fast MA', linewidth=1.5, alpha=0.7)
        if 'slow_ma' in signals.columns:
            ax1.plot(signals.index, signals['slow_ma'], label='Slow MA', linewidth=1.5, alpha=0.7)
        
        # Mark entry/exit points
        if 'entry_long' in signals.columns:
            long_entries = signals[signals['entry_long'] == True]
            if not long_entries.empty:
                ax1.scatter(long_entries.index, data.loc[long_entries.index, 'close'], 
                           color=self.colors['success'], marker='^', s=100, label='Long Entry', zorder=5)
        
        if 'entry_short' in signals.columns:
            short_entries = signals[signals['entry_short'] == True]
            if not short_entries.empty:
                ax1.scatter(short_entries.index, data.loc[short_entries.index, 'close'], 
                           color=self.colors['danger'], marker='v', s=100, label='Short Entry', zorder=5)
        
        ax1.set_title('Price Action and Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2 = axes[1]
        if 'rsi' in signals.columns:
            ax2.plot(signals.index, signals['rsi'], label='RSI', color=self.colors['primary'], linewidth=2)
            ax2.axhline(y=70, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color=self.colors['success'], linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.fill_between(signals.index, 70, 100, alpha=0.2, color=self.colors['danger'])
            ax2.fill_between(signals.index, 0, 30, alpha=0.2, color=self.colors['success'])
        
        ax2.set_title('RSI Oscillator')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volume
        ax3 = axes[2]
        if 'volume' in data.columns:
            ax3.bar(data.index, data['volume'], alpha=0.6, color=self.colors['info'])
            ax3.set_title('Volume')
            ax3.set_ylabel('Volume')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Signals
        ax4 = axes[3]
        if 'signal' in signals.columns:
            signal_values = signals['signal'].fillna(0)
            colors = [self.colors['danger'] if x < 0 else self.colors['success'] if x > 0 else self.colors['light'] 
                     for x in signal_values]
            
            ax4.bar(signals.index, signal_values, color=colors, alpha=0.7, width=1)
            ax4.axhline(y=0, color='black', linewidth=1)
            ax4.set_title('Trading Signals')
            ax4.set_ylabel('Signal')
            ax4.set_xlabel('Date')
            ax4.set_ylim(-1.5, 1.5)
            ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy overview chart saved to {save_path}")
        
        plt.show()
    
    def plot_equity_curve(self,
                         equity_data: pd.DataFrame,
                         benchmark_data: Optional[pd.DataFrame] = None,
                         drawdown_data: Optional[pd.Series] = None,
                         title: str = "Equity Curve",
                         save_path: Optional[str] = None) -> None:
        """
        Plot equity curve with drawdown.
        
        Args:
            equity_data: Equity curve data
            benchmark_data: Benchmark data for comparison
            drawdown_data: Drawdown series
            title: Chart title
            save_path: Path to save chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot equity curve
        if 'portfolio_value' in equity_data.columns:
            ax1.plot(equity_data.index, equity_data['portfolio_value'], 
                    label='Strategy', linewidth=2, color=self.colors['primary'])
        
        if benchmark_data is not None and 'portfolio_value' in benchmark_data.columns:
            ax1.plot(benchmark_data.index, benchmark_data['portfolio_value'], 
                    label='Benchmark', linewidth=2, color=self.colors['secondary'], alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        if drawdown_data is not None:
            ax2.fill_between(drawdown_data.index, drawdown_data * 100, 0, 
                           color=self.colors['danger'], alpha=0.3, label='Drawdown')
            ax2.plot(drawdown_data.index, drawdown_data * 100, 
                    color=self.colors['danger'], linewidth=1)
        
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve chart saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self,
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                title: str = "Returns Distribution",
                                save_path: Optional[str] = None) -> None:
        """
        Plot returns distribution analysis.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            title: Chart title
            save_path: Path to save chart
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Histogram
        ax1 = axes[0, 0]
        ax1.hist(returns * 100, bins=50, alpha=0.7, color=self.colors['primary'], 
                label='Strategy', density=True)
        if benchmark_returns is not None:
            ax1.hist(benchmark_returns * 100, bins=50, alpha=0.5, color=self.colors['secondary'], 
                    label='Benchmark', density=True)
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q Plot
        ax2 = axes[0, 1]
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Volatility
        ax3 = axes[1, 0]
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        ax3.plot(rolling_vol.index, rolling_vol, color=self.colors['warning'], linewidth=2)
        ax3.set_title('30-Day Rolling Volatility')
        ax3.set_ylabel('Volatility (%)')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative Returns
        ax4 = axes[1, 1]
        cumulative = (1 + returns).cumprod() - 1
        ax4.plot(cumulative.index, cumulative * 100, color=self.colors['success'], linewidth=2)
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            ax4.plot(benchmark_cumulative.index, benchmark_cumulative * 100, 
                    color=self.colors['secondary'], linewidth=2, alpha=0.7)
        ax4.set_title('Cumulative Returns')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution chart saved to {save_path}")
        
        plt.show()
    
    def plot_trade_analysis(self,
                          trades: List[Dict],
                          title: str = "Trade Analysis",
                          save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive trade analysis.
        
        Args:
            trades: List of trade dictionaries
            title: Chart title
            save_path: Path to save chart
        """
        if not trades:
            logger.warning("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract trade data
        pnls = [trade.get('pnl', 0) for trade in trades]
        durations = [trade.get('duration_days', 0) for trade in trades]
        entry_dates = [trade.get('entry_time') for trade in trades if trade.get('entry_time')]
        
        # PnL Distribution
        ax1 = axes[0, 0]
        colors = [self.colors['success'] if pnl > 0 else self.colors['danger'] for pnl in pnls]
        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_title('Trade PnL Distribution')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('PnL ($)')
        ax1.grid(True, alpha=0.3)
        
        # PnL Histogram
        ax2 = axes[0, 1]
        ax2.hist(pnls, bins=20, alpha=0.7, color=self.colors['info'])
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('PnL Histogram')
        ax2.set_xlabel('PnL ($)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Trade Duration
        ax3 = axes[1, 0]
        ax3.hist(durations, bins=20, alpha=0.7, color=self.colors['warning'])
        ax3.set_title('Trade Duration Distribution')
        ax3.set_xlabel('Duration (Days)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative PnL
        ax4 = axes[1, 1]
        cumulative_pnl = np.cumsum(pnls)
        ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                color=self.colors['primary'], linewidth=2, marker='o', markersize=4)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.set_title('Cumulative PnL')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative PnL ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade analysis chart saved to {save_path}")
        
        plt.show()
    
    def plot_risk_metrics(self,
                         risk_data: pd.DataFrame,
                         title: str = "Risk Metrics Over Time",
                         save_path: Optional[str] = None) -> None:
        """
        Plot risk metrics over time.
        
        Args:
            risk_data: Risk metrics data
            title: Chart title
            save_path: Path to save chart
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # VaR
        ax1 = axes[0, 0]
        if 'var_95' in risk_data.columns:
            ax1.plot(risk_data.index, risk_data['var_95'] * 100, 
                    color=self.colors['danger'], linewidth=2)
            ax1.set_title('Value at Risk (95%)')
            ax1.set_ylabel('VaR (%)')
            ax1.grid(True, alpha=0.3)
        
        # Volatility
        ax2 = axes[0, 1]
        if 'volatility' in risk_data.columns:
            ax2.plot(risk_data.index, risk_data['volatility'] * 100, 
                    color=self.colors['warning'], linewidth=2)
            ax2.set_title('Rolling Volatility')
            ax2.set_ylabel('Volatility (%)')
            ax2.grid(True, alpha=0.3)
        
        # Sharpe Ratio
        ax3 = axes[1, 0]
        if 'sharpe_ratio' in risk_data.columns:
            ax3.plot(risk_data.index, risk_data['sharpe_ratio'], 
                    color=self.colors['success'], linewidth=2)
            ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Rolling Sharpe Ratio')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
        
        # Max Drawdown
        ax4 = axes[1, 1]
        if 'max_drawdown' in risk_data.columns:
            ax4.plot(risk_data.index, risk_data['max_drawdown'] * 100, 
                    color=self.colors['danger'], linewidth=2)
            ax4.set_title('Rolling Max Drawdown')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk metrics chart saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self,
                                   data: pd.DataFrame,
                                   signals: pd.DataFrame,
                                   equity_curve: pd.DataFrame,
                                   trades: Optional[List[Dict]] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard.
        
        Args:
            data: OHLCV data
            signals: Strategy signals
            equity_curve: Equity curve data
            trades: Trade data
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price & Signals', 'RSI', 'Equity Curve', 'Volume'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.25, 0.15]
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], name='Close Price', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if 'fast_ma' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['fast_ma'], name='Fast MA',
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'slow_ma' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['slow_ma'], name='Slow MA',
                          line=dict(color='red', width=1)),
                row=1, col=1
            )
        
        # Entry points
        if 'entry_long' in signals.columns:
            long_entries = signals[signals['entry_long'] == True]
            if not long_entries.empty:
                fig.add_trace(
                    go.Scatter(x=long_entries.index, 
                             y=data.loc[long_entries.index, 'close'],
                             mode='markers', name='Long Entry',
                             marker=dict(symbol='triangle-up', size=10, color='green')),
                    row=1, col=1
                )
        
        # RSI
        if 'rsi' in signals.columns:
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['rsi'], name='RSI',
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Equity curve
        if 'portfolio_value' in equity_curve.columns:
            fig.add_trace(
                go.Scatter(x=equity_curve.index, y=equity_curve['portfolio_value'],
                          name='Portfolio Value', line=dict(color='blue', width=2)),
                row=3, col=1
            )
        
        # Volume
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(x=data.index, y=data['volume'], name='Volume',
                      marker_color='lightblue', opacity=0.6),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Interactive Strategy Dashboard",
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        return fig
