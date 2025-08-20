"""
Basic CTA Strategy Example

This example demonstrates how to:
1. Fetch market data
2. Create a simple CTA strategy
3. Run a backtest
4. Analyze performance
5. Visualize results
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import get_stock_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
from analytics import PerformanceAnalyzer, PerformanceReporter
from visualization import StrategyVisualizer
import pandas as pd

def main():
    """Run basic strategy example."""
    print("🚀 Basic CTA Strategy Example")
    print("="*50)
    
    # 1. Fetch market data
    print("\n📊 Step 1: Fetching market data...")
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    data = get_stock_data(symbol, start_date=start_date, end_date=end_date)
    print(f"✅ Fetched {len(data)} days of data for {symbol}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 2. Create strategy
    print("\n🎯 Step 2: Creating CTA strategy...")
    strategy = CTAStrategy(
        fast_ma_period=10,      # Fast moving average period
        slow_ma_period=30,      # Slow moving average period
        rsi_period=14,          # RSI period
        rsi_overbought=70,      # RSI overbought threshold
        rsi_oversold=30,        # RSI oversold threshold
        stop_loss_pct=0.02,     # 2% stop loss
        take_profit_pct=0.06,   # 6% take profit
        position_size_pct=0.1,  # 10% position size
        initial_capital=100000  # $100,000 starting capital
    )
    
    print("✅ Strategy created with parameters:")
    print(f"   Fast MA: {strategy.fast_ma_period} days")
    print(f"   Slow MA: {strategy.slow_ma_period} days")
    print(f"   RSI: {strategy.rsi_period} days")
    print(f"   Stop Loss: {strategy.stop_loss_pct:.1%}")
    print(f"   Take Profit: {strategy.take_profit_pct:.1%}")
    
    # 3. Generate signals
    print("\n📈 Step 3: Generating trading signals...")
    signals = strategy.generate_signals(data, symbol)
    
    # Count signals
    long_entries = signals['entry_long'].sum() if 'entry_long' in signals.columns else 0
    short_entries = signals['entry_short'].sum() if 'entry_short' in signals.columns else 0
    
    print(f"✅ Generated signals for {len(signals)} days:")
    print(f"   Long entry signals: {long_entries}")
    print(f"   Short entry signals: {short_entries}")
    
    # 4. Run backtest
    print("\n🔬 Step 4: Running backtest...")
    backtester = StrategyBacktester(
        strategy=strategy,
        initial_capital=100000,
        commission_rate=0.001  # 0.1% commission
    )
    
    results = backtester.run_backtest({symbol: data})
    
    print("✅ Backtest completed:")
    print(f"   Initial capital: ${results['summary']['initial_capital']:,.2f}")
    print(f"   Final portfolio value: ${results['summary']['final_portfolio_value']:,.2f}")
    print(f"   Total return: {results['summary']['total_return']:.2%}")
    print(f"   Total trades: {len(results['trades'])}")
    
    # 5. Analyze performance
    print("\n📊 Step 5: Analyzing performance...")
    
    # Calculate returns
    if not results['equity_curve'].empty:
        equity_values = results['equity_curve']['portfolio_value']
        returns = equity_values.pct_change().dropna()
        
        # Performance metrics
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(returns, trades=results['trades'])
        
        print("✅ Performance metrics calculated:")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"   Maximum Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Volatility: {metrics.volatility:.2%}")
        print(f"   Win Rate: {metrics.win_rate:.1f}%")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        
        # Generate comprehensive report
        reporter = PerformanceReporter()
        report = reporter.generate_summary_report(
            returns=returns,
            trades=results['trades'],
            strategy_name=f"{symbol} CTA Strategy"
        )
        
        # Save report
        os.makedirs('results', exist_ok=True)
        reporter.export_report(report, 'results/basic_strategy_report.json', 'json')
        print("✅ Performance report saved to results/basic_strategy_report.json")
        
        # 6. Visualize results
        print("\n🎨 Step 6: Creating visualizations...")
        
        visualizer = StrategyVisualizer()
        
        # Strategy overview chart
        visualizer.plot_strategy_overview(
            data=data,
            signals=signals,
            trades=results['trades'],
            title=f"{symbol} CTA Strategy Overview",
            save_path='results/basic_strategy_overview.png'
        )
        
        # Equity curve
        drawdown = (equity_values - equity_values.expanding().max()) / equity_values.expanding().max()
        visualizer.plot_equity_curve(
            equity_data=results['equity_curve'],
            drawdown_data=drawdown,
            title=f"{symbol} Portfolio Performance",
            save_path='results/basic_equity_curve.png'
        )
        
        # Returns distribution
        visualizer.plot_returns_distribution(
            returns=returns,
            title=f"{symbol} Returns Analysis",
            save_path='results/basic_returns_distribution.png'
        )
        
        # Trade analysis
        if results['trades']:
            visualizer.plot_trade_analysis(
                trades=results['trades'],
                title=f"{symbol} Trade Analysis",
                save_path='results/basic_trade_analysis.png'
            )
        
        print("✅ Visualizations created and saved to results/ directory")
        
        # 7. Summary
        print("\n🎯 Step 7: Summary")
        print("="*50)
        print(f"Strategy Performance Summary for {symbol}:")
        print(f"  📈 Total Return: {metrics.total_return:.2%}")
        print(f"  📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  📉 Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  🎯 Win Rate: {metrics.win_rate:.1f}%")
        print(f"  💰 Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  🔄 Total Trades: {len(results['trades'])}")
        
        # Performance rating
        if 'performance_rating' in report:
            rating = report['performance_rating']
            print(f"  ⭐ Performance Rating: {rating['rating']} ({rating['percentage']:.0f}/100)")
        
        print(f"\n📁 Generated Files:")
        print(f"  - results/basic_strategy_report.json")
        print(f"  - results/basic_strategy_overview.png")
        print(f"  - results/basic_equity_curve.png")
        print(f"  - results/basic_returns_distribution.png")
        if results['trades']:
            print(f"  - results/basic_trade_analysis.png")
        
        print(f"\n💡 Next Steps:")
        print(f"  - Try different parameters")
        print(f"  - Test on different assets")
        print(f"  - Compare with benchmark")
        print(f"  - Optimize parameters")
        print(f"  - Add risk management rules")
        
    else:
        print("❌ No equity curve data available")
    
    print("\n🎉 Basic strategy example completed!")

if __name__ == "__main__":
    main()
