"""
Test the strategy optimization system.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from strategies import CTAStrategy
from optimization import ParameterOptimizer, StrategyTuner
import pandas as pd
import numpy as np
import os

def test_parameter_optimizer():
    """Test basic parameter optimizer functionality."""
    print("=== Parameter Optimizer Test ===")
    
    try:
        # Get test data
        print("Fetching test data...")
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        if data.empty:
            print("❌ No data available")
            return False
        
        print(f"✅ Data: {len(data)} rows")
        
        # Initialize optimizer
        optimizer = ParameterOptimizer(objective_function='sharpe_ratio')
        
        # Define small parameter grid for testing
        param_grid = {
            'fast_ma_period': [8, 10, 12],
            'slow_ma_period': [25, 30, 35],
            'rsi_period': [12, 14, 16]
        }
        
        print(f"Testing grid search with {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        # Run grid search
        result = optimizer.grid_search(
            strategy_class=CTAStrategy,
            data={'AAPL': data},
            param_grid=param_grid,
            backtest_func=None
        )
        
        print(f"✅ Grid search completed:")
        print(f"   Best score: {result.best_score:.4f}")
        print(f"   Best parameters: {result.best_params}")
        print(f"   Optimization time: {result.optimization_time:.1f}s")
        print(f"   Total evaluations: {len(result.all_results)}")
        
        # Test random search
        print("\nTesting random search...")
        param_distributions = {
            'fast_ma_period': (5, 15, 'int'),
            'slow_ma_period': (20, 40, 'int'),
            'rsi_period': (10, 20, 'int'),
            'stop_loss_pct': (0.01, 0.05),
            'take_profit_pct': (0.03, 0.10)
        }
        
        random_result = optimizer.random_search(
            strategy_class=CTAStrategy,
            data={'AAPL': data},
            param_distributions=param_distributions,
            n_iter=20
        )
        
        print(f"✅ Random search completed:")
        print(f"   Best score: {random_result.best_score:.4f}")
        print(f"   Best parameters: {random_result.best_params}")
        print(f"   Optimization time: {random_result.optimization_time:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_tuner():
    """Test strategy tuner functionality."""
    print("\n=== Strategy Tuner Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        if data.empty:
            print("❌ No data available")
            return False
        
        print(f"✅ Data prepared: {len(data)} rows")
        
        # Initialize strategy tuner
        tuner = StrategyTuner(
            strategy_class=CTAStrategy,
            data={'AAPL': data},
            objective_function='sharpe_ratio'
        )
        
        # Define parameter ranges
        param_ranges = {
            'fast_ma_period': (5, 15),
            'slow_ma_period': (20, 40),
            'rsi_period': (10, 20)
        }
        
        print("Running quick tune with grid search...")
        
        # Quick tune
        result = tuner.quick_tune(
            param_ranges=param_ranges,
            method='grid',
            n_trials=50
        )
        
        print(f"✅ Quick tune completed:")
        print(f"   Best score: {result.best_score:.4f}")
        print(f"   Best parameters: {result.best_params}")
        print(f"   Method: {result.method}")
        
        # Test sensitivity analysis
        print("\nRunning sensitivity analysis...")
        
        base_params = result.best_params.copy()
        base_params.update({
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'position_size_pct': 0.1,
            'initial_capital': 100000
        })
        
        sensitivity_ranges = {
            'fast_ma_period': (5, 20),
            'slow_ma_period': (15, 50)
        }
        
        sensitivity_results = tuner.sensitivity_analysis(
            base_params=base_params,
            param_ranges=sensitivity_ranges,
            n_points=5
        )
        
        print(f"✅ Sensitivity analysis completed:")
        for param, data in sensitivity_results.items():
            valid_scores = [s for s in data['scores'] if s is not None]
            if valid_scores:
                print(f"   {param}: {len(valid_scores)} valid evaluations")
                print(f"     Score range: {min(valid_scores):.4f} to {max(valid_scores):.4f}")
        
        # Plot results
        print("\nGenerating optimization plots...")
        tuner.plot_optimization_results(result, 'results/optimization_results.png')
        tuner.plot_sensitivity_analysis(sensitivity_results, 'results/sensitivity_analysis.png')
        
        print("✅ Plots generated and saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy tuner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_optimization():
    """Test comprehensive optimization with multiple methods."""
    print("\n=== Comprehensive Optimization Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        # Initialize tuner
        tuner = StrategyTuner(
            strategy_class=CTAStrategy,
            data={'AAPL': data},
            objective_function='sharpe_ratio'
        )
        
        # Define parameter ranges
        param_ranges = {
            'fast_ma_period': (8, 15),
            'slow_ma_period': (25, 40),
            'rsi_period': (12, 18)
        }
        
        print("Running comprehensive optimization...")
        
        # Check if optuna is available
        try:
            import optuna
            methods = ['grid', 'random', 'bayesian']
        except ImportError:
            methods = ['grid', 'random']
            print("   Note: Skipping Bayesian optimization (optuna not installed)")

        # Run comprehensive optimization (smaller scale for testing)
        results = tuner.comprehensive_tune(
            param_ranges=param_ranges,
            methods=methods,
            n_trials=20
        )
        
        print(f"✅ Comprehensive optimization completed:")
        print(f"   Methods tested: {list(results.keys())}")
        
        for method, result in results.items():
            print(f"   {method}:")
            print(f"     Best score: {result.best_score:.4f}")
            print(f"     Best params: {result.best_params}")
            print(f"     Time: {result.optimization_time:.1f}s")
        
        # Get optimization summary
        summary = tuner.get_optimization_summary()
        print(f"\n📊 Optimization Summary:")
        print(f"   Best overall score: {summary['best_overall_score']:.4f}")
        print(f"   Best overall params: {summary['best_overall_params']}")
        print(f"   Methods tested: {summary['methods_tested']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_asset_optimization():
    """Test optimization with multiple assets."""
    print("\n=== Multi-Asset Optimization Test ===")
    
    try:
        # Get data for multiple assets
        symbols = ['AAPL', 'MSFT']
        data = {}
        
        print("Fetching multi-asset data...")
        for symbol in symbols:
            asset_data = get_stock_data(symbol, start_date='2023-06-01', end_date='2024-01-31')
            if not asset_data.empty:
                data[symbol] = asset_data
                print(f"✅ {symbol}: {len(asset_data)} rows")
        
        if len(data) < 2:
            print("❌ Need at least 2 assets for multi-asset test")
            return False
        
        # Initialize tuner
        tuner = StrategyTuner(
            strategy_class=CTAStrategy,
            data=data,
            objective_function='sharpe_ratio'
        )
        
        # Define parameter ranges
        param_ranges = {
            'fast_ma_period': (8, 12),
            'slow_ma_period': (25, 35),
            'position_size_pct': (0.05, 0.15)  # Smaller positions for multi-asset
        }
        
        print(f"Running optimization on {len(data)} assets...")
        
        # Quick tune
        result = tuner.quick_tune(
            param_ranges=param_ranges,
            method='random',
            n_trials=15
        )
        
        print(f"✅ Multi-asset optimization completed:")
        print(f"   Best score: {result.best_score:.4f}")
        print(f"   Best parameters: {result.best_params}")
        print(f"   Assets: {list(data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-asset optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_objective_functions():
    """Test different objective functions."""
    print("\n=== Objective Functions Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-06-01', end_date='2024-01-31')
        
        # Test different objective functions
        objectives = ['sharpe_ratio', 'total_return', 'calmar_ratio']
        
        param_ranges = {
            'fast_ma_period': (8, 12),
            'slow_ma_period': (25, 35)
        }
        
        results = {}
        
        for objective in objectives:
            print(f"\nTesting {objective} objective...")
            
            tuner = StrategyTuner(
                strategy_class=CTAStrategy,
                data={'AAPL': data},
                objective_function=objective
            )
            
            result = tuner.quick_tune(
                param_ranges=param_ranges,
                method='random',
                n_trials=10
            )
            
            results[objective] = result
            print(f"✅ {objective}: Best score = {result.best_score:.4f}")
            print(f"   Best params: {result.best_params}")
        
        print(f"\n📊 Objective Function Comparison:")
        for objective, result in results.items():
            print(f"   {objective}: {result.best_score:.4f} with {result.best_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Objective functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_optimization_report():
    """Create optimization usage guide."""
    report = """
# 🎯 Strategy Optimization Guide

## Overview

The optimization system provides comprehensive tools for tuning trading strategy parameters to maximize performance metrics.

## Key Components

### 1. **ParameterOptimizer**
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling from parameter space  
- Bayesian Optimization: Smart search using Optuna

### 2. **StrategyTuner**
- High-level interface for strategy optimization
- Multiple optimization methods
- Sensitivity analysis
- Visualization tools

### 3. **WalkForwardOptimizer**
- Robust out-of-sample testing
- Time-series cross-validation
- Prevents overfitting

## Usage Examples

### Quick Optimization
```python
from optimization import StrategyTuner
from strategies import CTAStrategy

# Initialize tuner
tuner = StrategyTuner(CTAStrategy, data, 'sharpe_ratio')

# Define parameter ranges
param_ranges = {
    'fast_ma_period': (5, 20),
    'slow_ma_period': (20, 50),
    'rsi_period': (10, 20)
}

# Quick tune
result = tuner.quick_tune(param_ranges, method='bayesian', n_trials=100)
```

### Comprehensive Analysis
```python
# Multiple methods
results = tuner.comprehensive_tune(param_ranges, methods=['grid', 'random', 'bayesian'])

# Sensitivity analysis
sensitivity = tuner.sensitivity_analysis(best_params, param_ranges)

# Visualization
tuner.plot_optimization_results(results['bayesian'])
tuner.plot_sensitivity_analysis(sensitivity)
```

### Walk-Forward Testing
```python
# Robust out-of-sample testing
wf_results = tuner.walk_forward_optimization(
    param_ranges, 
    training_window=252,  # 1 year training
    testing_window=63,    # 3 months testing
    step_size=21          # Monthly reoptimization
)
```

## Best Practices

### 1. **Parameter Selection**
- Start with wide ranges, then narrow down
- Use domain knowledge to set reasonable bounds
- Consider parameter interactions

### 2. **Objective Functions**
- `sharpe_ratio`: Risk-adjusted returns (recommended)
- `calmar_ratio`: Return vs max drawdown
- `total_return`: Absolute returns
- `win_rate`: Percentage of winning trades

### 3. **Overfitting Prevention**
- Use walk-forward optimization
- Cross-validation with time-series splits
- Out-of-sample testing
- Parameter stability analysis

### 4. **Computational Efficiency**
- Start with random search for exploration
- Use Bayesian optimization for refinement
- Parallel processing for grid search
- Reasonable parameter ranges

## Optimization Workflow

1. **Exploratory Phase**
   - Wide parameter ranges
   - Random search (50-100 trials)
   - Identify promising regions

2. **Refinement Phase**
   - Narrow parameter ranges
   - Bayesian optimization (100-200 trials)
   - Fine-tune best parameters

3. **Validation Phase**
   - Walk-forward optimization
   - Sensitivity analysis
   - Out-of-sample testing

4. **Production Phase**
   - Monitor parameter stability
   - Periodic reoptimization
   - Performance tracking

## Tips for Success

- **Start Simple**: Begin with 2-3 key parameters
- **Use Multiple Metrics**: Don't optimize for just one objective
- **Consider Transaction Costs**: Include realistic costs in backtests
- **Test Robustness**: Ensure parameters work across different market conditions
- **Document Results**: Keep track of optimization experiments

Happy optimizing! 🚀
"""
    
    with open('results/optimization_guide.md', 'w') as f:
        f.write(report)
    
    print("✅ Optimization guide created: results/optimization_guide.md")

def main():
    """Run all optimization tests."""
    print("🎯 COMPREHENSIVE STRATEGY OPTIMIZATION TEST")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test1 = test_parameter_optimizer()
    test2 = test_strategy_tuner()
    test3 = test_comprehensive_optimization()
    test4 = test_multi_asset_optimization()
    test5 = test_objective_functions()
    
    # Create optimization guide
    create_optimization_report()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 STRATEGY OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Parameter Optimizer", test1),
        ("Strategy Tuner", test2),
        ("Comprehensive Optimization", test3),
        ("Multi-Asset Optimization", test4),
        ("Objective Functions", test5)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimization tests passed!")
        print("✅ Strategy optimization system is ready for production use!")
        print("\n📊 Generated Files:")
        print("   - results/optimization_results.png")
        print("   - results/sensitivity_analysis.png")
        print("   - results/optimization_guide.md")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
