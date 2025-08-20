"""
Strategy Optimization Example

This example demonstrates how to:
1. Set up parameter optimization
2. Run different optimization methods
3. Analyze optimization results
4. Perform sensitivity analysis
5. Validate optimized parameters
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import get_stock_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
from analytics import PerformanceAnalyzer
import pandas as pd
import numpy as np

def manual_optimization_example():
    """Example of manual parameter optimization."""
    print("🎯 Manual Parameter Optimization Example")
    print("="*50)
    
    # Get data
    print("\n📊 Fetching data...")
    data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
    print(f"✅ Data: {len(data)} days")
    
    # Define parameter combinations to test
    param_combinations = [
        {'fast_ma_period': 8, 'slow_ma_period': 25, 'rsi_period': 14},
        {'fast_ma_period': 10, 'slow_ma_period': 30, 'rsi_period': 14},
        {'fast_ma_period': 12, 'slow_ma_period': 35, 'rsi_period': 14},
        {'fast_ma_period': 10, 'slow_ma_period': 25, 'rsi_period': 12},
        {'fast_ma_period': 10, 'slow_ma_period': 30, 'rsi_period': 16},
        {'fast_ma_period': 15, 'slow_ma_period': 40, 'rsi_period': 14},
        {'fast_ma_period': 8, 'slow_ma_period': 30, 'rsi_period': 12},
        {'fast_ma_period': 12, 'slow_ma_period': 25, 'rsi_period': 16}
    ]
    
    print(f"\n🔍 Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i, params in enumerate(param_combinations):
        try:
            # Create strategy with parameters
            strategy = CTAStrategy(
                fast_ma_period=params['fast_ma_period'],
                slow_ma_period=params['slow_ma_period'],
                rsi_period=params['rsi_period'],
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                position_size_pct=0.1,
                initial_capital=100000
            )
            
            # Run backtest
            backtester = StrategyBacktester(strategy=strategy, initial_capital=100000)
            backtest_results = backtester.run_backtest({'AAPL': data})
            
            # Calculate performance
            if not backtest_results['equity_curve'].empty:
                equity_values = backtest_results['equity_curve']['portfolio_value']
                returns = equity_values.pct_change().dropna()
                
                if len(returns) > 0:
                    analyzer = PerformanceAnalyzer()
                    metrics = analyzer.calculate_metrics(returns)
                    
                    result = {
                        'params': params,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'total_return': metrics.total_return,
                        'max_drawdown': metrics.max_drawdown,
                        'volatility': metrics.volatility,
                        'win_rate': metrics.win_rate,
                        'total_trades': len(backtest_results['trades'])
                    }
                    
                    results.append(result)
                    
                    if metrics.sharpe_ratio > best_score:
                        best_score = metrics.sharpe_ratio
                        best_params = params
                    
                    print(f"   {i+1:2d}. Fast MA: {params['fast_ma_period']:2d}, "
                          f"Slow MA: {params['slow_ma_period']:2d}, "
                          f"RSI: {params['rsi_period']:2d} → "
                          f"Sharpe: {metrics.sharpe_ratio:6.3f}, "
                          f"Return: {metrics.total_return:7.2%}")
        
        except Exception as e:
            print(f"   {i+1:2d}. Error: {e}")
    
    # Display results
    if results:
        print(f"\n✅ Optimization completed!")
        print(f"   Best Sharpe ratio: {best_score:.3f}")
        print(f"   Best parameters: {best_params}")
        
        # Sort results by Sharpe ratio
        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        print(f"\n🏆 Top 5 parameter combinations:")
        print(f"{'Rank':<4} {'Fast MA':<7} {'Slow MA':<7} {'RSI':<4} {'Sharpe':<7} {'Return':<8} {'Max DD':<8}")
        print("-" * 60)
        
        for i, result in enumerate(results[:5]):
            params = result['params']
            print(f"{i+1:<4} {params['fast_ma_period']:<7} {params['slow_ma_period']:<7} "
                  f"{params['rsi_period']:<4} {result['sharpe_ratio']:<7.3f} "
                  f"{result['total_return']:<8.2%} {result['max_drawdown']:<8.2%}")
        
        return results, best_params
    
    else:
        print("❌ No valid results found")
        return [], None

def sensitivity_analysis_example(best_params):
    """Example of parameter sensitivity analysis."""
    print(f"\n🔬 Parameter Sensitivity Analysis")
    print("="*50)
    
    if not best_params:
        print("❌ No best parameters available for sensitivity analysis")
        return
    
    # Get data
    data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
    
    # Test sensitivity of fast MA period
    print(f"\n📊 Testing fast_ma_period sensitivity around {best_params['fast_ma_period']}...")
    
    base_params = best_params.copy()
    fast_ma_values = range(max(5, best_params['fast_ma_period'] - 3), 
                          best_params['fast_ma_period'] + 4)
    
    sensitivity_results = []
    
    for fast_ma in fast_ma_values:
        try:
            test_params = base_params.copy()
            test_params['fast_ma_period'] = fast_ma
            
            strategy = CTAStrategy(
                fast_ma_period=test_params['fast_ma_period'],
                slow_ma_period=test_params['slow_ma_period'],
                rsi_period=test_params['rsi_period'],
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                position_size_pct=0.1,
                initial_capital=100000
            )
            
            backtester = StrategyBacktester(strategy=strategy, initial_capital=100000)
            backtest_results = backtester.run_backtest({'AAPL': data})
            
            if not backtest_results['equity_curve'].empty:
                equity_values = backtest_results['equity_curve']['portfolio_value']
                returns = equity_values.pct_change().dropna()
                
                if len(returns) > 0:
                    analyzer = PerformanceAnalyzer()
                    metrics = analyzer.calculate_metrics(returns)
                    
                    sensitivity_results.append({
                        'fast_ma_period': fast_ma,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'total_return': metrics.total_return,
                        'max_drawdown': metrics.max_drawdown
                    })
                    
                    marker = "★" if fast_ma == best_params['fast_ma_period'] else " "
                    print(f"   {marker} Fast MA {fast_ma:2d}: Sharpe = {metrics.sharpe_ratio:6.3f}, "
                          f"Return = {metrics.total_return:7.2%}, DD = {metrics.max_drawdown:7.2%}")
        
        except Exception as e:
            print(f"   Fast MA {fast_ma:2d}: Error - {e}")
    
    if sensitivity_results:
        # Find best and worst in sensitivity test
        best_sensitivity = max(sensitivity_results, key=lambda x: x['sharpe_ratio'])
        worst_sensitivity = min(sensitivity_results, key=lambda x: x['sharpe_ratio'])
        
        print(f"\n✅ Sensitivity Analysis Results:")
        print(f"   Best in range: Fast MA {best_sensitivity['fast_ma_period']} "
              f"(Sharpe: {best_sensitivity['sharpe_ratio']:.3f})")
        print(f"   Worst in range: Fast MA {worst_sensitivity['fast_ma_period']} "
              f"(Sharpe: {worst_sensitivity['sharpe_ratio']:.3f})")
        
        # Calculate sensitivity metrics
        sharpe_values = [r['sharpe_ratio'] for r in sensitivity_results]
        sharpe_std = np.std(sharpe_values)
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        
        print(f"   Sharpe ratio std dev: {sharpe_std:.3f}")
        print(f"   Sharpe ratio range: {sharpe_range:.3f}")
        
        if sharpe_std < 0.1:
            print(f"   ✅ Parameter appears stable (low sensitivity)")
        else:
            print(f"   ⚠️ Parameter appears sensitive (high variability)")
    
    return sensitivity_results

def walk_forward_validation_example(best_params):
    """Example of walk-forward validation."""
    print(f"\n🚶 Walk-Forward Validation Example")
    print("="*50)
    
    if not best_params:
        print("❌ No best parameters available for validation")
        return
    
    # Get longer data period for walk-forward
    data = get_stock_data('AAPL', start_date='2022-01-01', end_date='2024-01-31')
    
    if len(data) < 500:
        print("❌ Insufficient data for walk-forward validation")
        return
    
    print(f"✅ Using {len(data)} days of data for walk-forward validation")
    
    # Define walk-forward parameters
    training_window = 252  # 1 year training
    testing_window = 63    # 3 months testing
    step_size = 21         # Monthly rebalancing
    
    print(f"   Training window: {training_window} days")
    print(f"   Testing window: {testing_window} days")
    print(f"   Step size: {step_size} days")
    
    # Perform walk-forward validation
    validation_results = []
    current_start = 0
    
    while current_start + training_window + testing_window <= len(data):
        try:
            # Define periods
            train_end = current_start + training_window
            test_start = train_end
            test_end = test_start + testing_window
            
            # Split data
            train_data = data.iloc[current_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Test on out-of-sample period
            strategy = CTAStrategy(
                fast_ma_period=best_params['fast_ma_period'],
                slow_ma_period=best_params['slow_ma_period'],
                rsi_period=best_params['rsi_period'],
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                position_size_pct=0.1,
                initial_capital=100000
            )
            
            backtester = StrategyBacktester(strategy=strategy, initial_capital=100000)
            test_results = backtester.run_backtest({'AAPL': test_data})
            
            if not test_results['equity_curve'].empty:
                equity_values = test_results['equity_curve']['portfolio_value']
                returns = equity_values.pct_change().dropna()
                
                if len(returns) > 0:
                    analyzer = PerformanceAnalyzer()
                    metrics = analyzer.calculate_metrics(returns)
                    
                    period_result = {
                        'period': len(validation_results) + 1,
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': test_data.index[0],
                        'test_end': test_data.index[-1],
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'total_return': metrics.total_return,
                        'max_drawdown': metrics.max_drawdown,
                        'trades': len(test_results['trades'])
                    }
                    
                    validation_results.append(period_result)
                    
                    print(f"   Period {len(validation_results):2d}: "
                          f"{test_data.index[0].strftime('%Y-%m-%d')} to "
                          f"{test_data.index[-1].strftime('%Y-%m-%d')} → "
                          f"Sharpe: {metrics.sharpe_ratio:6.3f}, "
                          f"Return: {metrics.total_return:7.2%}")
        
        except Exception as e:
            print(f"   Period error: {e}")
        
        # Move to next period
        current_start += step_size
    
    if validation_results:
        print(f"\n✅ Walk-forward validation completed ({len(validation_results)} periods)")
        
        # Calculate summary statistics
        sharpe_ratios = [r['sharpe_ratio'] for r in validation_results]
        returns = [r['total_return'] for r in validation_results]
        drawdowns = [r['max_drawdown'] for r in validation_results]
        
        print(f"\n📊 Validation Summary:")
        print(f"   Average Sharpe ratio: {np.mean(sharpe_ratios):6.3f} ± {np.std(sharpe_ratios):6.3f}")
        print(f"   Average return: {np.mean(returns):7.2%} ± {np.std(returns):7.2%}")
        print(f"   Average max drawdown: {np.mean(drawdowns):7.2%} ± {np.std(drawdowns):7.2%}")
        print(f"   Positive periods: {sum(1 for r in returns if r > 0)}/{len(returns)} "
              f"({sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%)")
        
        # Stability assessment
        sharpe_stability = np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf')
        
        print(f"\n🎯 Stability Assessment:")
        print(f"   Sharpe ratio coefficient of variation: {sharpe_stability:.3f}")
        
        if sharpe_stability < 0.5:
            print(f"   ✅ Strategy appears stable across periods")
        elif sharpe_stability < 1.0:
            print(f"   ⚠️ Strategy shows moderate variability")
        else:
            print(f"   ❌ Strategy shows high variability (potentially overfit)")
    
    return validation_results

def main():
    """Run optimization example."""
    print("🎯 STRATEGY OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 1. Manual optimization
    optimization_results, best_params = manual_optimization_example()
    
    # 2. Sensitivity analysis
    if best_params:
        sensitivity_results = sensitivity_analysis_example(best_params)
    
    # 3. Walk-forward validation
    if best_params:
        validation_results = walk_forward_validation_example(best_params)
    
    # 4. Summary and recommendations
    print(f"\n🎯 OPTIMIZATION SUMMARY")
    print("="*60)
    
    if best_params:
        print(f"✅ Optimization completed successfully!")
        print(f"\n🏆 Best Parameters Found:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['sharpe_ratio'])
            print(f"\n📊 Best Performance:")
            print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
            print(f"   Total Return: {best_result['total_return']:.2%}")
            print(f"   Max Drawdown: {best_result['max_drawdown']:.2%}")
            print(f"   Total Trades: {best_result['total_trades']}")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. Use the optimized parameters as a starting point")
        print(f"   2. Test on different assets and time periods")
        print(f"   3. Consider transaction costs in optimization")
        print(f"   4. Monitor parameter stability over time")
        print(f"   5. Implement walk-forward optimization in production")
        
        # Save results
        if optimization_results:
            results_df = pd.DataFrame(optimization_results)
            results_df.to_csv('results/optimization_results.csv', index=False)
            print(f"\n📁 Results saved to results/optimization_results.csv")
    
    else:
        print(f"❌ Optimization failed - no valid results found")
    
    print(f"\n🎉 Optimization example completed!")

if __name__ == "__main__":
    main()
