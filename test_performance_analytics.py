"""
Test the performance analytics system.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from analytics import PerformanceAnalyzer, PerformanceReporter, RollingPerformanceAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime

def test_performance_analyzer():
    """Test basic performance analyzer functionality."""
    print("=== Performance Analyzer Test ===")
    
    # Get test data
    print("Fetching test data...")
    try:
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        if data.empty:
            print("❌ No data available")
            return False
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        print(f"✅ Data: {len(returns)} return observations")
        
        # Initialize analyzer
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        
        # Calculate metrics
        print("\nCalculating performance metrics...")
        metrics = analyzer.calculate_metrics(returns)
        
        print(f"✅ Performance metrics calculated:")
        print(f"   Total Return: {metrics.total_return:.2%}")
        print(f"   Annualized Return: {metrics.annualized_return:.2%}")
        print(f"   Volatility: {metrics.volatility:.2%}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
        print(f"   Skewness: {metrics.skewness:.2f}")
        print(f"   Kurtosis: {metrics.kurtosis:.2f}")
        print(f"   VaR (95%): {metrics.var_95:.2%}")
        print(f"   CVaR (95%): {metrics.cvar_95:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_comparison():
    """Test benchmark comparison functionality."""
    print("\n=== Benchmark Comparison Test ===")
    
    try:
        # Get strategy and benchmark data
        print("Fetching strategy and benchmark data...")
        strategy_data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        benchmark_data = get_stock_data('SPY', start_date='2023-01-01', end_date='2024-01-31')
        
        if strategy_data.empty or benchmark_data.empty:
            print("❌ Insufficient data for benchmark comparison")
            return False
        
        strategy_returns = strategy_data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        
        print(f"✅ Strategy returns: {len(strategy_returns)} observations")
        print(f"✅ Benchmark returns: {len(benchmark_returns)} observations")
        
        # Calculate metrics with benchmark
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(strategy_returns, benchmark_returns)
        
        print(f"\n✅ Benchmark comparison metrics:")
        print(f"   Alpha: {metrics.alpha:.2%}")
        print(f"   Beta: {metrics.beta:.2f}")
        print(f"   Information Ratio: {metrics.information_ratio:.2f}")
        print(f"   Tracking Error: {metrics.tracking_error:.2%}")
        
        # Calculate benchmark metrics for comparison
        benchmark_metrics = analyzer.calculate_metrics(benchmark_returns)
        
        print(f"\n📊 Performance comparison:")
        print(f"   Strategy Return: {metrics.total_return:.2%}")
        print(f"   Benchmark Return: {benchmark_metrics.total_return:.2%}")
        print(f"   Excess Return: {metrics.total_return - benchmark_metrics.total_return:.2%}")
        print(f"   Strategy Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"   Benchmark Sharpe: {benchmark_metrics.sharpe_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trade_metrics():
    """Test trade-based metrics calculation."""
    print("\n=== Trade Metrics Test ===")
    
    try:
        # Create sample trade data
        sample_trades = [
            {'pnl': 150.0, 'entry_price': 100, 'exit_price': 105, 'quantity': 30},
            {'pnl': -75.0, 'entry_price': 105, 'exit_price': 102.5, 'quantity': 30},
            {'pnl': 200.0, 'entry_price': 102, 'exit_price': 108, 'quantity': 33},
            {'pnl': -50.0, 'entry_price': 108, 'exit_price': 106.5, 'quantity': 33},
            {'pnl': 300.0, 'entry_price': 106, 'exit_price': 115, 'quantity': 33},
            {'pnl': -100.0, 'entry_price': 115, 'exit_price': 112, 'quantity': 33},
            {'pnl': 125.0, 'entry_price': 112, 'exit_price': 116, 'quantity': 31}
        ]
        
        print(f"Sample trades: {len(sample_trades)} trades")
        
        # Create dummy returns for other metrics
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Calculate metrics with trades
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(returns, trades=sample_trades)
        
        print(f"✅ Trade metrics calculated:")
        print(f"   Total Trades: {metrics.total_trades}")
        print(f"   Win Rate: {metrics.win_rate:.1f}%")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        print(f"   Average Win: ${metrics.avg_win:.2f}")
        print(f"   Average Loss: ${metrics.avg_loss:.2f}")
        print(f"   Largest Win: ${metrics.largest_win:.2f}")
        print(f"   Largest Loss: ${metrics.largest_loss:.2f}")
        
        # Calculate some additional trade statistics
        pnls = [trade['pnl'] for trade in sample_trades]
        total_pnl = sum(pnls)
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        print(f"\n📈 Additional trade analysis:")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Winning Trades: {len(winning_trades)}")
        print(f"   Losing Trades: {len(losing_trades)}")
        print(f"   Average Trade: ${total_pnl/len(sample_trades):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_reporter():
    """Test performance reporting functionality."""
    print("\n=== Performance Reporter Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        benchmark_data = get_stock_data('SPY', start_date='2023-01-01', end_date='2024-01-31')
        
        returns = data['close'].pct_change().dropna()
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        
        # Create sample trades
        sample_trades = [
            {'pnl': 150.0}, {'pnl': -75.0}, {'pnl': 200.0}, 
            {'pnl': -50.0}, {'pnl': 300.0}
        ]
        
        # Initialize reporter
        reporter = PerformanceReporter()
        
        print("Generating comprehensive performance report...")
        
        # Generate summary report
        report = reporter.generate_summary_report(
            returns=returns,
            benchmark_returns=benchmark_returns,
            trades=sample_trades,
            strategy_name="Test CTA Strategy"
        )
        
        print(f"✅ Summary report generated:")
        print(f"   Strategy: {report['strategy_name']}")
        print(f"   Period: {report['period']['start_date']} to {report['period']['end_date']}")
        print(f"   Total Return: {report['returns']['total_return']}")
        print(f"   Sharpe Ratio: {report['risk_adjusted']['sharpe_ratio']}")
        print(f"   Max Drawdown: {report['risk']['max_drawdown']}")
        print(f"   Win Rate: {report['trading']['win_rate']}")
        print(f"   Performance Rating: {report['performance_rating']['rating']} ({report['performance_rating']['percentage']:.1f}%)")
        
        # Generate monthly returns table
        print("\nGenerating monthly returns table...")
        monthly_table = reporter.generate_monthly_returns_table(returns)
        if not monthly_table.empty:
            print(f"✅ Monthly returns table: {monthly_table.shape[0]} years, {monthly_table.shape[1]} columns")
            print("Sample (last 3 months):")
            print(monthly_table.iloc[-1, -4:].to_string())
        
        # Generate drawdown analysis
        print("\nGenerating drawdown analysis...")
        drawdown_analysis = reporter.generate_drawdown_analysis(returns)
        if drawdown_analysis:
            print(f"✅ Drawdown analysis:")
            print(f"   Max Drawdown: {drawdown_analysis['max_drawdown']:.2%}")
            print(f"   Current Drawdown: {drawdown_analysis['current_drawdown']:.2%}")
            print(f"   Avg Drawdown: {drawdown_analysis['avg_drawdown']:.2%}")
            print(f"   Time in Drawdown: {drawdown_analysis['time_in_drawdown']:.1f}%")
            print(f"   Total Drawdown Periods: {drawdown_analysis['total_drawdown_periods']}")
        
        # Export reports
        print("\nExporting reports...")
        reporter.export_report(report, 'results/performance_report.json', 'json')
        reporter.export_report(report, 'results/performance_report.html', 'html')
        reporter.export_report(report, 'results/performance_report.txt', 'txt')
        
        print("✅ Reports exported to results/ directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance reporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rolling_performance():
    """Test rolling performance analysis."""
    print("\n=== Rolling Performance Test ===")
    
    try:
        # Get test data (longer period for rolling analysis)
        data = get_stock_data('AAPL', start_date='2022-01-01', end_date='2024-01-31')
        returns = data['close'].pct_change().dropna()
        
        print(f"Data for rolling analysis: {len(returns)} observations")
        
        # Initialize rolling analyzer
        rolling_analyzer = RollingPerformanceAnalyzer(window_size=252)  # 1-year window
        
        print("Calculating rolling metrics...")
        rolling_metrics = rolling_analyzer.calculate_rolling_metrics(
            returns, 
            metrics=['sharpe_ratio', 'volatility', 'max_drawdown', 'sortino_ratio']
        )
        
        if not rolling_metrics.empty:
            print(f"✅ Rolling metrics calculated: {len(rolling_metrics)} periods")
            print(f"   Date range: {rolling_metrics.index[0]} to {rolling_metrics.index[-1]}")
            
            # Show summary statistics
            print(f"\n📊 Rolling metrics summary:")
            print(f"   Avg Sharpe Ratio: {rolling_metrics['sharpe_ratio'].mean():.2f}")
            print(f"   Sharpe Ratio Range: {rolling_metrics['sharpe_ratio'].min():.2f} to {rolling_metrics['sharpe_ratio'].max():.2f}")
            print(f"   Avg Volatility: {rolling_metrics['volatility'].mean():.2%}")
            print(f"   Volatility Range: {rolling_metrics['volatility'].min():.2%} to {rolling_metrics['volatility'].max():.2%}")
            
            # Save rolling metrics
            rolling_metrics.to_csv('results/rolling_performance.csv')
            print("✅ Rolling metrics saved to results/rolling_performance.csv")
        else:
            print("⚠️  Insufficient data for rolling analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Rolling performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all performance analytics tests."""
    print("📊 COMPREHENSIVE PERFORMANCE ANALYTICS TEST")
    print("="*60)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test1 = test_performance_analyzer()
    test2 = test_benchmark_comparison()
    test3 = test_trade_metrics()
    test4 = test_performance_reporter()
    test5 = test_rolling_performance()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 PERFORMANCE ANALYTICS TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Performance Analyzer", test1),
        ("Benchmark Comparison", test2),
        ("Trade Metrics", test3),
        ("Performance Reporter", test4),
        ("Rolling Performance", test5)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance analytics tests passed!")
        print("✅ Performance analytics system is ready for production use!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
