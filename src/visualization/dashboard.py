"""
Interactive dashboard for strategy monitoring and analysis.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import get_stock_data, get_crypto_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
from analytics import PerformanceAnalyzer, PerformanceReporter
from risk import RiskManager, PortfolioRiskMonitor


class StrategyDashboard:
    """
    Interactive Streamlit dashboard for strategy analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.setup_page_config()
        
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="CTA Strategy Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-metric {
            border-left-color: #28a745;
        }
        .warning-metric {
            border-left-color: #ffc107;
        }
        .danger-metric {
            border-left-color: #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the main dashboard."""
        st.markdown('<h1 class="main-header">🚀 CTA Strategy Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar for controls
        self.render_sidebar()
        
        # Main content
        if 'run_analysis' in st.session_state and st.session_state.run_analysis:
            self.render_main_content()
        else:
            self.render_welcome_page()
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("📊 Analysis Configuration")
        
        # Asset selection
        asset_type = st.sidebar.selectbox(
            "Asset Type",
            ["Stock", "Cryptocurrency"],
            help="Choose the type of asset to analyze"
        )
        
        if asset_type == "Stock":
            symbol = st.sidebar.selectbox(
                "Stock Symbol",
                ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
                help="Select a stock symbol"
            )
        else:
            symbol = st.sidebar.selectbox(
                "Crypto Pair",
                ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"],
                help="Select a cryptocurrency pair"
            )
        
        # Date range
        st.sidebar.subheader("📅 Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Start date for analysis"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End date for analysis"
            )
        
        # Strategy parameters
        st.sidebar.subheader("⚙️ Strategy Parameters")
        
        fast_ma = st.sidebar.slider("Fast MA Period", 5, 50, 10, help="Fast moving average period")
        slow_ma = st.sidebar.slider("Slow MA Period", 20, 100, 30, help="Slow moving average period")
        rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14, help="RSI calculation period")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            stop_loss = st.sidebar.number_input("Stop Loss %", 0.5, 10.0, 2.0, 0.1, help="Stop loss percentage")
        with col2:
            take_profit = st.sidebar.number_input("Take Profit %", 1.0, 20.0, 6.0, 0.1, help="Take profit percentage")
        
        position_size = st.sidebar.slider("Position Size %", 1, 20, 10, help="Position size as % of capital")
        
        # Risk management
        st.sidebar.subheader("🛡️ Risk Management")
        initial_capital = st.sidebar.number_input("Initial Capital", 10000, 1000000, 100000, 10000)
        commission = st.sidebar.number_input("Commission %", 0.0, 1.0, 0.1, 0.01, help="Commission rate")
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            st.session_state.run_analysis = True
            st.session_state.config = {
                'asset_type': asset_type,
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi_period': rsi_period,
                'stop_loss': stop_loss / 100,
                'take_profit': take_profit / 100,
                'position_size': position_size / 100,
                'initial_capital': initial_capital,
                'commission': commission / 100
            }
            st.rerun()
    
    def render_welcome_page(self):
        """Render the welcome page."""
        st.markdown("""
        ## Welcome to the CTA Strategy Dashboard! 👋
        
        This interactive dashboard allows you to:
        
        ### 📈 **Strategy Analysis**
        - Backtest CTA strategies on stocks and cryptocurrencies
        - Analyze performance metrics and risk indicators
        - Visualize trading signals and portfolio evolution
        
        ### 🛡️ **Risk Management**
        - Monitor portfolio risk in real-time
        - Track drawdowns and volatility
        - Set custom risk parameters
        
        ### 📊 **Performance Analytics**
        - Compare against benchmarks
        - Analyze trade statistics
        - Generate comprehensive reports
        
        ### 🎯 **Getting Started**
        1. Configure your analysis parameters in the sidebar
        2. Select an asset and date range
        3. Adjust strategy parameters
        4. Click "Run Analysis" to start
        
        ---
        
        **Ready to begin?** Configure your settings in the sidebar and click "Run Analysis"!
        """)
        
        # Show sample metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assets Supported", "9+", "Stocks & Crypto")
        
        with col2:
            st.metric("Indicators", "5+", "MA, RSI, MACD")
        
        with col3:
            st.metric("Risk Metrics", "15+", "VaR, Sharpe, etc.")
        
        with col4:
            st.metric("Timeframes", "Multiple", "1h, 1d, 1w")
    
    def render_main_content(self):
        """Render the main analysis content."""
        config = st.session_state.config
        
        # Load data and run analysis
        with st.spinner("Loading data and running analysis..."):
            try:
                results = self.run_strategy_analysis(config)
                
                if results is None:
                    st.error("Failed to run analysis. Please check your parameters and try again.")
                    return
                
                # Display results
                self.display_performance_overview(results)
                self.display_charts(results)
                self.display_detailed_metrics(results)
                self.display_trade_analysis(results)
                self.display_risk_analysis(results)
                
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)
    
    def run_strategy_analysis(self, config: Dict) -> Optional[Dict]:
        """Run the strategy analysis."""
        try:
            # Fetch data
            if config['asset_type'] == 'Stock':
                data = get_stock_data(
                    config['symbol'],
                    start_date=config['start_date'],
                    end_date=config['end_date']
                )
            else:
                data = get_crypto_data(
                    config['symbol'],
                    start_date=config['start_date'],
                    end_date=config['end_date']
                )
            
            if data.empty:
                st.error("No data available for the selected parameters.")
                return None
            
            # Initialize strategy
            strategy = CTAStrategy(
                fast_ma_period=config['fast_ma'],
                slow_ma_period=config['slow_ma'],
                rsi_period=config['rsi_period'],
                stop_loss_pct=config['stop_loss'],
                take_profit_pct=config['take_profit'],
                position_size_pct=config['position_size'],
                initial_capital=config['initial_capital']
            )
            
            # Run backtest
            backtester = StrategyBacktester(
                strategy=strategy,
                initial_capital=config['initial_capital'],
                commission_rate=config['commission']
            )
            
            backtest_results = backtester.run_backtest({config['symbol']: data})
            
            # Generate signals for visualization
            signals = strategy.generate_signals(data, config['symbol'])
            
            # Calculate performance metrics
            if backtest_results['equity_curve'].empty:
                returns = pd.Series(dtype=float)
            else:
                equity_values = backtest_results['equity_curve']['portfolio_value']
                returns = equity_values.pct_change().dropna()
            
            analyzer = PerformanceAnalyzer()
            performance_metrics = analyzer.calculate_metrics(returns, trades=backtest_results['trades'])
            
            return {
                'data': data,
                'signals': signals,
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics,
                'config': config
            }
            
        except Exception as e:
            st.error(f"Error in strategy analysis: {str(e)}")
            return None
    
    def display_performance_overview(self, results: Dict):
        """Display performance overview metrics."""
        st.header("📊 Performance Overview")
        
        metrics = results['performance_metrics']
        backtest = results['backtest_results']
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_return = metrics.total_return * 100
            color = "success" if total_return > 0 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h4>Total Return</h4>
                <h2>{total_return:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe = metrics.sharpe_ratio
            color = "success" if sharpe > 1 else "warning" if sharpe > 0 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h4>Sharpe Ratio</h4>
                <h2>{sharpe:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_dd = metrics.max_drawdown * 100
            color = "success" if max_dd > -10 else "warning" if max_dd > -20 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h4>Max Drawdown</h4>
                <h2>{max_dd:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            win_rate = metrics.win_rate
            color = "success" if win_rate > 60 else "warning" if win_rate > 40 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h4>Win Rate</h4>
                <h2>{win_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            total_trades = metrics.total_trades
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Trades</h4>
                <h2>{total_trades}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    def display_charts(self, results: Dict):
        """Display interactive charts."""
        st.header("📈 Strategy Visualization")
        
        data = results['data']
        signals = results['signals']
        equity_curve = results['backtest_results']['equity_curve']
        
        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Strategy Overview", "Equity Curve", "Returns Analysis"])
        
        with tab1:
            self.plot_strategy_overview(data, signals)
        
        with tab2:
            self.plot_equity_curve(equity_curve)
        
        with tab3:
            self.plot_returns_analysis(results)
    
    def plot_strategy_overview(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Plot strategy overview chart."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Signals', 'RSI', 'Trading Signals'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and moving averages
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
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Trading signals
        if 'signal' in signals.columns:
            signal_colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' 
                           for x in signals['signal'].fillna(0)]
            fig.add_trace(
                go.Bar(x=signals.index, y=signals['signal'].fillna(0),
                      name='Signals', marker_color=signal_colors, opacity=0.7),
                row=3, col=1
            )
        
        fig.update_layout(height=800, title="Strategy Overview")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Signal", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_equity_curve(self, equity_curve: pd.DataFrame):
        """Plot equity curve."""
        if equity_curve.empty:
            st.warning("No equity curve data available.")
            return
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve['portfolio_value'],
                      name='Portfolio Value', line=dict(color='blue', width=2))
        )
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_returns_analysis(self, results: Dict):
        """Plot returns analysis."""
        equity_curve = results['backtest_results']['equity_curve']
        
        if equity_curve.empty:
            st.warning("No returns data available.")
            return
        
        returns = equity_curve['portfolio_value'].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns histogram
            fig1 = px.histogram(x=returns * 100, nbins=30, 
                              title="Daily Returns Distribution",
                              labels={'x': 'Daily Returns (%)', 'y': 'Frequency'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Cumulative returns
            cumulative = (1 + returns).cumprod() - 1
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative * 100,
                          name='Cumulative Returns', line=dict(color='green', width=2))
            )
            fig2.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def display_detailed_metrics(self, results: Dict):
        """Display detailed performance metrics."""
        st.header("📋 Detailed Metrics")
        
        metrics = results['performance_metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Return Metrics")
            st.metric("Total Return", f"{metrics.total_return:.2%}")
            st.metric("Annualized Return", f"{metrics.annualized_return:.2%}")
            st.metric("CAGR", f"{metrics.cagr:.2%}")
        
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Volatility", f"{metrics.volatility:.2%}")
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
            st.metric("VaR (95%)", f"{metrics.var_95:.2%}")
        
        with col3:
            st.subheader("Risk-Adjusted Returns")
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
            st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
    
    def display_trade_analysis(self, results: Dict):
        """Display trade analysis."""
        st.header("💼 Trade Analysis")
        
        trades = results['backtest_results']['trades']
        
        if not trades:
            st.warning("No trades executed in this backtest.")
            return
        
        # Trade statistics
        pnls = [trade['pnl'] for trade in trades]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades))
        
        with col2:
            winning_trades = len([pnl for pnl in pnls if pnl > 0])
            st.metric("Winning Trades", winning_trades)
        
        with col3:
            avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0
            st.metric("Avg Win", f"${avg_win:.2f}")
        
        with col4:
            avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        
        # Trade PnL chart
        fig = px.bar(x=range(len(pnls)), y=pnls, 
                    color=[pnl > 0 for pnl in pnls],
                    title="Trade PnL Distribution",
                    labels={'x': 'Trade Number', 'y': 'PnL ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_analysis(self, results: Dict):
        """Display risk analysis."""
        st.header("🛡️ Risk Analysis")
        
        metrics = results['performance_metrics']
        
        # Risk metrics table
        risk_data = {
            'Metric': ['Value at Risk (95%)', 'Conditional VaR (95%)', 'Skewness', 'Kurtosis', 'Ulcer Index'],
            'Value': [f"{metrics.var_95:.2%}", f"{metrics.cvar_95:.2%}", 
                     f"{metrics.skewness:.2f}", f"{metrics.kurtosis:.2f}", f"{metrics.ulcer_index:.2f}"]
        }
        
        st.table(pd.DataFrame(risk_data))


def main():
    """Main function to run the dashboard."""
    dashboard = StrategyDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
