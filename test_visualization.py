"""
Test the visualization and dashboard system.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
from visualization import StrategyVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def test_strategy_visualizer():
    """Test the strategy visualizer."""
    print("=== Strategy Visualizer Test ===")
    
    try:
        # Get test data
        print("Fetching test data...")
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        if data.empty:
            print("❌ No data available")
            return False
        
        print(f"✅ Data: {len(data)} rows")
        
        # Generate strategy signals
        strategy = CTAStrategy()
        signals = strategy.generate_signals(data, 'AAPL')
        
        print(f"✅ Signals generated: {len(signals)} rows")
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Test strategy overview chart
        print("\nTesting strategy overview chart...")
        visualizer.plot_strategy_overview(
            data=data,
            signals=signals,
            title="AAPL CTA Strategy Test",
            save_path="results/strategy_overview_test.png"
        )
        print("✅ Strategy overview chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_equity_curve_visualization():
    """Test equity curve visualization."""
    print("\n=== Equity Curve Visualization Test ===")
    
    try:
        # Get test data and run backtest
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        strategy = CTAStrategy(initial_capital=100000)
        backtester = StrategyBacktester(strategy=strategy, initial_capital=100000)
        
        print("Running backtest for visualization...")
        results = backtester.run_backtest({'AAPL': data})
        
        if results['equity_curve'].empty:
            print("❌ No equity curve data")
            return False
        
        print(f"✅ Backtest completed: {len(results['equity_curve'])} equity points")
        
        # Calculate drawdown
        equity_values = results['equity_curve']['portfolio_value']
        running_max = equity_values.expanding().max()
        drawdown = (equity_values - running_max) / running_max
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Test equity curve chart
        print("Creating equity curve chart...")
        visualizer.plot_equity_curve(
            equity_data=results['equity_curve'],
            drawdown_data=drawdown,
            title="AAPL Portfolio Equity Curve",
            save_path="results/equity_curve_test.png"
        )
        print("✅ Equity curve chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Equity curve visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_returns_distribution():
    """Test returns distribution visualization."""
    print("\n=== Returns Distribution Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        benchmark_data = get_stock_data('SPY', start_date='2023-01-01', end_date='2024-01-31')
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        
        print(f"✅ Returns calculated: {len(returns)} strategy, {len(benchmark_returns)} benchmark")
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Test returns distribution chart
        print("Creating returns distribution chart...")
        visualizer.plot_returns_distribution(
            returns=returns,
            benchmark_returns=benchmark_returns,
            title="AAPL vs SPY Returns Distribution",
            save_path="results/returns_distribution_test.png"
        )
        print("✅ Returns distribution chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Returns distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trade_analysis_visualization():
    """Test trade analysis visualization."""
    print("\n=== Trade Analysis Visualization Test ===")
    
    try:
        # Create sample trade data
        sample_trades = [
            {'pnl': 150.0, 'duration_days': 5, 'entry_time': '2023-01-15', 'exit_time': '2023-01-20'},
            {'pnl': -75.0, 'duration_days': 3, 'entry_time': '2023-01-25', 'exit_time': '2023-01-28'},
            {'pnl': 200.0, 'duration_days': 7, 'entry_time': '2023-02-01', 'exit_time': '2023-02-08'},
            {'pnl': -50.0, 'duration_days': 2, 'entry_time': '2023-02-15', 'exit_time': '2023-02-17'},
            {'pnl': 300.0, 'duration_days': 10, 'entry_time': '2023-02-20', 'exit_time': '2023-03-02'},
            {'pnl': -100.0, 'duration_days': 4, 'entry_time': '2023-03-10', 'exit_time': '2023-03-14'},
            {'pnl': 125.0, 'duration_days': 6, 'entry_time': '2023-03-20', 'exit_time': '2023-03-26'},
            {'pnl': -80.0, 'duration_days': 3, 'entry_time': '2023-04-01', 'exit_time': '2023-04-04'},
            {'pnl': 175.0, 'duration_days': 8, 'entry_time': '2023-04-10', 'exit_time': '2023-04-18'},
            {'pnl': 90.0, 'duration_days': 4, 'entry_time': '2023-04-25', 'exit_time': '2023-04-29'}
        ]
        
        print(f"✅ Sample trades created: {len(sample_trades)} trades")
        
        # Calculate trade statistics
        pnls = [trade['pnl'] for trade in sample_trades]
        total_pnl = sum(pnls)
        winning_trades = len([pnl for pnl in pnls if pnl > 0])
        win_rate = winning_trades / len(sample_trades) * 100
        
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Winning Trades: {winning_trades}/{len(sample_trades)}")
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Test trade analysis chart
        print("Creating trade analysis chart...")
        visualizer.plot_trade_analysis(
            trades=sample_trades,
            title="Sample Trade Analysis",
            save_path="results/trade_analysis_test.png"
        )
        print("✅ Trade analysis chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade analysis visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_dashboard():
    """Test interactive dashboard creation."""
    print("\n=== Interactive Dashboard Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        # Generate signals and run backtest
        strategy = CTAStrategy()
        signals = strategy.generate_signals(data, 'AAPL')
        
        backtester = StrategyBacktester(strategy=strategy)
        results = backtester.run_backtest({'AAPL': data})
        
        print(f"✅ Data prepared for interactive dashboard")
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Create interactive dashboard
        print("Creating interactive dashboard...")
        fig = visualizer.create_interactive_dashboard(
            data=data,
            signals=signals,
            equity_curve=results['equity_curve']
        )
        
        # Save as HTML
        fig.write_html("results/interactive_dashboard_test.html")
        print("✅ Interactive dashboard created and saved as HTML")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_metrics_visualization():
    """Test risk metrics visualization."""
    print("\n=== Risk Metrics Visualization Test ===")
    
    try:
        # Create sample risk metrics data
        dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
        
        # Generate synthetic risk metrics
        np.random.seed(42)
        risk_data = pd.DataFrame({
            'var_95': np.random.normal(-0.02, 0.005, len(dates)),
            'volatility': np.random.normal(0.15, 0.03, len(dates)),
            'sharpe_ratio': np.random.normal(1.2, 0.3, len(dates)),
            'max_drawdown': np.random.normal(-0.08, 0.02, len(dates))
        }, index=dates)
        
        # Ensure realistic bounds
        risk_data['var_95'] = np.clip(risk_data['var_95'], -0.05, 0)
        risk_data['volatility'] = np.clip(risk_data['volatility'], 0.05, 0.5)
        risk_data['sharpe_ratio'] = np.clip(risk_data['sharpe_ratio'], -1, 3)
        risk_data['max_drawdown'] = np.clip(risk_data['max_drawdown'], -0.3, 0)
        
        print(f"✅ Risk metrics data created: {len(risk_data)} days")
        
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Test risk metrics chart
        print("Creating risk metrics chart...")
        visualizer.plot_risk_metrics(
            risk_data=risk_data,
            title="Sample Risk Metrics Over Time",
            save_path="results/risk_metrics_test.png"
        )
        print("✅ Risk metrics chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk metrics visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dashboard_instructions():
    """Create instructions for running the Streamlit dashboard."""
    instructions = """
# 🚀 CTA Strategy Dashboard Instructions

## Running the Interactive Dashboard

To run the interactive Streamlit dashboard:

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit plotly
   ```

2. **Run the dashboard**:
   ```bash
   streamlit run src/visualization/dashboard.py
   ```

3. **Access the dashboard**:
   - Open your browser to http://localhost:8501
   - The dashboard will load automatically

## Dashboard Features

### 📊 **Strategy Configuration**
- Select asset type (Stock or Cryptocurrency)
- Choose from popular symbols
- Set date range for analysis
- Adjust strategy parameters (MA periods, RSI, stop loss, etc.)
- Configure risk management settings

### 📈 **Interactive Visualizations**
- Strategy overview with price, signals, and indicators
- Equity curve with drawdown analysis
- Returns distribution and analysis
- Trade analysis and statistics

### 🛡️ **Risk Analysis**
- Real-time risk metrics
- Portfolio risk monitoring
- Performance vs benchmark comparison

### 📋 **Performance Reports**
- Comprehensive performance metrics
- Trade statistics and analysis
- Risk-adjusted returns
- Detailed analytics

## Usage Tips

1. **Start Simple**: Begin with default parameters and a familiar stock like AAPL
2. **Experiment**: Try different parameter combinations to see their impact
3. **Compare Assets**: Test the same strategy on different assets
4. **Analyze Results**: Use the detailed metrics to understand performance drivers
5. **Export Data**: Charts and data can be downloaded for further analysis

## Troubleshooting

- **Data Issues**: Ensure you have internet connection for data fetching
- **Performance**: Large date ranges may take longer to process
- **Browser**: Use Chrome or Firefox for best compatibility

Enjoy exploring your CTA strategy performance! 🎯
"""
    
    with open('results/dashboard_instructions.md', 'w') as f:
        f.write(instructions)
    
    print("✅ Dashboard instructions created: results/dashboard_instructions.md")

def main():
    """Run all visualization tests."""
    print("🎨 COMPREHENSIVE VISUALIZATION SYSTEM TEST")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test1 = test_strategy_visualizer()
    test2 = test_equity_curve_visualization()
    test3 = test_returns_distribution()
    test4 = test_trade_analysis_visualization()
    test5 = test_interactive_dashboard()
    test6 = test_risk_metrics_visualization()
    
    # Create dashboard instructions
    create_dashboard_instructions()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 VISUALIZATION SYSTEM TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Strategy Visualizer", test1),
        ("Equity Curve Visualization", test2),
        ("Returns Distribution", test3),
        ("Trade Analysis Visualization", test4),
        ("Interactive Dashboard", test5),
        ("Risk Metrics Visualization", test6)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All visualization tests passed!")
        print("✅ Visualization system is ready for production use!")
        print("\n📊 Generated Files:")
        print("   - results/strategy_overview_test.png")
        print("   - results/equity_curve_test.png") 
        print("   - results/returns_distribution_test.png")
        print("   - results/trade_analysis_test.png")
        print("   - results/interactive_dashboard_test.html")
        print("   - results/risk_metrics_test.png")
        print("   - results/dashboard_instructions.md")
        print("\n🚀 To run the interactive dashboard:")
        print("   streamlit run src/visualization/dashboard.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
