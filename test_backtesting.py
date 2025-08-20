"""
Test the backtesting framework with CTA strategy.
"""
import sys
sys.path.append('src')

from data import get_stock_data, get_crypto_data
from strategies import CTAStrategy
from backtesting import StrategyBacktester
import pandas as pd
import json

def test_single_asset_backtest():
    """Test backtesting on a single asset."""
    print("=== Single Asset Backtest Test ===")
    
    # Get test data
    print("Fetching AAPL data...")
    data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
    
    if data.empty:
        print("❌ No data available")
        return False
    
    print(f"✅ Data: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Initialize strategy
    strategy = CTAStrategy(
        fast_ma_period=10,
        slow_ma_period=30,
        rsi_period=14,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        position_size_pct=0.1,
        initial_capital=100000
    )
    
    # Initialize backtester
    backtester = StrategyBacktester(
        strategy=strategy,
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    print("Running backtest...")
    try:
        results = backtester.run_backtest(
            data={'AAPL': data},
            warmup_period=50
        )
        
        print("✅ Backtest completed!")
        
        # Display results
        summary = results['summary']
        performance = results['performance_metrics']
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"   Final Value: ${summary['final_portfolio_value']:,.2f}")
        print(f"   Total Return: {summary['total_return']*100:.2f}%")
        print(f"   Annualized Return: {performance['annualized_return']*100:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']*100:.2f}%")
        print(f"   Total Commission: ${summary['total_commission']:.2f}")
        print(f"   Total Slippage: ${summary['total_slippage']:.2f}")
        
        # Show sample trades
        if results['trades']:
            print(f"\n📈 SAMPLE TRADES:")
            for i, trade in enumerate(results['trades'][:3]):
                pnl_pct = (trade['pnl'] / (trade['entry_price'] * trade['quantity'])) * 100
                print(f"   {i+1}. {trade['side']} {trade['symbol']}: "
                      f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} "
                      f"({trade['duration_days']}d), PnL: ${trade['pnl']:.2f} ({pnl_pct:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_asset_backtest():
    """Test backtesting on multiple assets."""
    print("\n=== Multi-Asset Backtest Test ===")
    
    # Get data for multiple assets
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = {}
    
    print("Fetching data for multiple assets...")
    for symbol in symbols:
        try:
            asset_data = get_stock_data(symbol, start_date='2023-06-01', end_date='2024-01-31')
            if not asset_data.empty:
                data[symbol] = asset_data
                print(f"✅ {symbol}: {len(asset_data)} rows")
            else:
                print(f"⚠️  {symbol}: No data")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    if not data:
        print("❌ No data available for multi-asset test")
        return False
    
    # Initialize strategy
    strategy = CTAStrategy(
        fast_ma_period=10,
        slow_ma_period=30,
        position_size_pct=0.05,  # Smaller position size for multiple assets
        initial_capital=100000
    )
    
    # Initialize backtester
    backtester = StrategyBacktester(
        strategy=strategy,
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    print(f"Running multi-asset backtest on {len(data)} assets...")
    try:
        results = backtester.run_backtest(
            data=data,
            warmup_period=50
        )
        
        print("✅ Multi-asset backtest completed!")
        
        # Display results
        summary = results['summary']
        performance = results['performance_metrics']
        
        print(f"\n📊 MULTI-ASSET RESULTS:")
        print(f"   Assets Traded: {len(data)}")
        print(f"   Total Return: {summary['total_return']*100:.2f}%")
        print(f"   Annualized Return: {performance['annualized_return']*100:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']*100:.2f}%")
        
        # Show trades by asset
        if results['trades']:
            trades_by_asset = {}
            for trade in results['trades']:
                symbol = trade['symbol']
                if symbol not in trades_by_asset:
                    trades_by_asset[symbol] = []
                trades_by_asset[symbol].append(trade)
            
            print(f"\n📈 TRADES BY ASSET:")
            for symbol, trades in trades_by_asset.items():
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
                print(f"   {symbol}: {len(trades)} trades, ${total_pnl:.2f} PnL, {win_rate:.1f}% win rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-asset backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_crypto_backtest():
    """Test backtesting on crypto asset."""
    print("\n=== Crypto Backtest Test ===")
    
    # Get crypto data
    print("Fetching BTC/USDT data...")
    try:
        data = get_crypto_data('BTC/USDT', start_date='2023-06-01', end_date='2024-01-31')
        
        if data.empty:
            print("❌ No crypto data available")
            return False
        
        print(f"✅ BTC/USDT: {len(data)} rows")
        
        # Initialize strategy with crypto-optimized parameters
        strategy = CTAStrategy(
            fast_ma_period=8,
            slow_ma_period=21,
            rsi_period=14,
            stop_loss_pct=0.03,  # Higher stop loss for crypto volatility
            take_profit_pct=0.08,  # Higher take profit
            position_size_pct=0.05,  # Smaller position size
            initial_capital=50000
        )
        
        # Initialize backtester
        backtester = StrategyBacktester(
            strategy=strategy,
            initial_capital=50000,
            commission_rate=0.001,
            slippage_rate=0.001  # Higher slippage for crypto
        )
        
        print("Running crypto backtest...")
        results = backtester.run_backtest(
            data={'BTC/USDT': data},
            warmup_period=30
        )
        
        print("✅ Crypto backtest completed!")
        
        # Display results
        summary = results['summary']
        performance = results['performance_metrics']
        
        print(f"\n₿ CRYPTO BACKTEST RESULTS:")
        print(f"   Total Return: {summary['total_return']*100:.2f}%")
        print(f"   Annualized Return: {performance['annualized_return']*100:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']*100:.1f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']*100:.2f}%")
        print(f"   Volatility: {performance['volatility']*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_backtest_results():
    """Save detailed backtest results to file."""
    print("\n=== Saving Detailed Results ===")
    
    # Run a comprehensive backtest and save results
    try:
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-31')
        
        strategy = CTAStrategy(initial_capital=100000)
        backtester = StrategyBacktester(strategy=strategy, initial_capital=100000)
        
        results = backtester.run_backtest(data={'AAPL': data})
        
        # Save results to JSON (excluding non-serializable objects)
        save_results = {
            'summary': results['summary'],
            'performance_metrics': results['performance_metrics'],
            'risk_metrics': results['risk_metrics'],
            'benchmark_comparison': results['benchmark_comparison'],
            'trades_count': len(results['trades']),
            'orders_count': len(results['orders'])
        }
        
        # Convert timestamps to strings
        if save_results['summary']['start_date']:
            save_results['summary']['start_date'] = save_results['summary']['start_date'].isoformat()
        if save_results['summary']['end_date']:
            save_results['summary']['end_date'] = save_results['summary']['end_date'].isoformat()
        
        with open('results/backtest_results.json', 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        print("✅ Results saved to results/backtest_results.json")
        
        # Save equity curve
        equity_df = results['equity_curve']
        equity_df.to_csv('results/equity_curve.csv', index=False)
        print("✅ Equity curve saved to results/equity_curve.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return False

def main():
    """Run all backtesting tests."""
    print("🚀 COMPREHENSIVE BACKTESTING FRAMEWORK TEST")
    print("="*60)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test1 = test_single_asset_backtest()
    test2 = test_multi_asset_backtest()
    test3 = test_crypto_backtest()
    test4 = save_backtest_results()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 BACKTESTING FRAMEWORK TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Single Asset Backtest", test1),
        ("Multi-Asset Backtest", test2),
        ("Crypto Backtest", test3),
        ("Results Saving", test4)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All backtesting tests passed!")
        print("✅ Backtesting framework is ready for production use!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
