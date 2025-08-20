"""
Comprehensive test of the CTA strategy with multiple assets and scenarios.
"""
import sys
sys.path.append('src')

from data import get_stock_data, get_crypto_data
from strategies import CTAStrategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_strategy_on_asset(symbol, data_fetcher, timeframe='1d', start_date='2023-01-01', end_date='2024-01-01'):
    """Test strategy on a single asset."""
    print(f"\n=== Testing {symbol} ===")
    
    # Fetch data
    try:
        data = data_fetcher(symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
        if data.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"✅ Data: {len(data)} rows from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   Volatility: {data['close'].pct_change().std()*100:.2f}% daily")
        
    except Exception as e:
        print(f"❌ Data fetch failed for {symbol}: {e}")
        return None
    
    # Initialize strategy
    strategy = CTAStrategy(
        fast_ma_period=10,
        slow_ma_period=30,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        ma_type='sma',
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        position_size_pct=0.1,
        initial_capital=100000
    )
    
    # Generate signals
    try:
        signals = strategy.generate_signals(data, symbol)
        
        # Count signals
        long_entries = signals['entry_long'].sum() if 'entry_long' in signals.columns else 0
        short_entries = signals['entry_short'].sum() if 'entry_short' in signals.columns else 0
        exit_signals = signals['exit_signal'].sum() if 'exit_signal' in signals.columns else 0
        
        print(f"✅ Signals: {long_entries} long, {short_entries} short, {exit_signals} exits")
        
        # Calculate signal frequency
        total_signals = long_entries + short_entries
        signal_frequency = total_signals / len(data) * 100 if len(data) > 0 else 0
        print(f"   Signal frequency: {signal_frequency:.1f}% of trading days")
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return None
    
    # Simulate trading
    try:
        # Combine data with signals for processing
        data_with_signals = data.copy()
        signal_cols = ['entry_long', 'entry_short', 'exit_signal', 'signal_strength']
        for col in signal_cols:
            if col in signals.columns:
                data_with_signals[col] = signals[col].fillna(False)
            else:
                data_with_signals[col] = False
        
        # Process all signals
        actions = strategy.process_signals(data_with_signals, symbol)
        
        print(f"✅ Trading simulation:")
        print(f"   Total actions: {len(actions)}")
        print(f"   Open positions: {len(strategy.get_current_positions())}")
        print(f"   Closed positions: {len(strategy.closed_positions)}")
        print(f"   Remaining capital: ${strategy.current_capital:.2f}")
        
        # Calculate performance metrics
        metrics = strategy.get_performance_metrics()
        if metrics:
            print(f"   Performance metrics:")
            print(f"     Total trades: {metrics.get('total_trades', 0)}")
            print(f"     Win rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"     Total PnL: ${metrics.get('total_pnl', 0):.2f}")
            print(f"     Avg PnL per trade: ${metrics.get('avg_pnl', 0):.2f}")
            print(f"     Max win: ${metrics.get('max_win', 0):.2f}")
            print(f"     Max loss: ${metrics.get('max_loss', 0):.2f}")
        
        # Show sample trades
        if strategy.closed_positions:
            print(f"   Sample trades:")
            for i, pos in enumerate(strategy.closed_positions[:3]):
                duration = (pos.exit_time - pos.entry_time).days
                print(f"     {i+1}. {pos.side} {pos.entry_time.strftime('%Y-%m-%d')} to {pos.exit_time.strftime('%Y-%m-%d')} "
                      f"({duration}d): ${pos.entry_price:.2f} -> ${pos.exit_price:.2f}, PnL: ${pos.pnl:.2f}")
        
        return {
            'symbol': symbol,
            'data_points': len(data),
            'signals': total_signals,
            'actions': len(actions),
            'closed_trades': len(strategy.closed_positions),
            'metrics': metrics,
            'final_capital': strategy.current_capital
        }
        
    except Exception as e:
        print(f"❌ Trading simulation failed: {e}")
        return None

def test_multiple_stocks():
    """Test strategy on multiple stock symbols."""
    print("\n" + "="*50)
    print("STOCK MARKET TESTING")
    print("="*50)
    
    # Popular stocks with different characteristics
    stocks = [
        'AAPL',  # Tech giant, relatively stable
        'TSLA',  # High volatility
        'SPY',   # Market index
        'GOOGL', # Another tech stock
        'MSFT'   # Stable tech
    ]
    
    results = []
    for stock in stocks:
        result = test_strategy_on_asset(stock, get_stock_data, 
                                      start_date='2023-06-01', end_date='2024-02-01')
        if result:
            results.append(result)
    
    return results

def test_multiple_crypto():
    """Test strategy on multiple crypto symbols."""
    print("\n" + "="*50)
    print("CRYPTOCURRENCY TESTING")
    print("="*50)
    
    # Popular crypto pairs
    cryptos = [
        'BTC/USDT',  # Bitcoin
        'ETH/USDT',  # Ethereum
        'ADA/USDT',  # Cardano
        'SOL/USDT'   # Solana
    ]
    
    results = []
    for crypto in cryptos:
        result = test_strategy_on_asset(crypto, get_crypto_data,
                                      start_date='2023-06-01', end_date='2024-02-01')
        if result:
            results.append(result)
    
    return results

def test_different_parameters():
    """Test strategy with different parameter sets."""
    print("\n" + "="*50)
    print("PARAMETER SENSITIVITY TESTING")
    print("="*50)
    
    # Get test data
    data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-01')
    if data.empty:
        print("❌ No data for parameter testing")
        return []
    
    print(f"Using AAPL data: {len(data)} rows")
    
    # Different parameter combinations
    param_sets = [
        {'name': 'Conservative', 'fast': 20, 'slow': 50, 'rsi': 14, 'stop': 0.015, 'profit': 0.045},
        {'name': 'Aggressive', 'fast': 5, 'slow': 15, 'rsi': 10, 'stop': 0.03, 'profit': 0.09},
        {'name': 'Balanced', 'fast': 10, 'slow': 30, 'rsi': 14, 'stop': 0.02, 'profit': 0.06},
        {'name': 'Trend Following', 'fast': 8, 'slow': 21, 'rsi': 21, 'stop': 0.025, 'profit': 0.075}
    ]
    
    results = []
    for params in param_sets:
        print(f"\n--- Testing {params['name']} Parameters ---")
        print(f"Fast MA: {params['fast']}, Slow MA: {params['slow']}, RSI: {params['rsi']}")
        print(f"Stop Loss: {params['stop']*100:.1f}%, Take Profit: {params['profit']*100:.1f}%")
        
        try:
            strategy = CTAStrategy(
                fast_ma_period=params['fast'],
                slow_ma_period=params['slow'],
                rsi_period=params['rsi'],
                stop_loss_pct=params['stop'],
                take_profit_pct=params['profit'],
                initial_capital=100000
            )
            
            # Generate signals and simulate
            signals = strategy.generate_signals(data, 'AAPL')
            
            # Combine data with signals
            data_with_signals = data.copy()
            signal_cols = ['entry_long', 'entry_short', 'exit_signal', 'signal_strength']
            for col in signal_cols:
                if col in signals.columns:
                    data_with_signals[col] = signals[col].fillna(False)
                else:
                    data_with_signals[col] = False
            
            actions = strategy.process_signals(data_with_signals, 'AAPL')
            metrics = strategy.get_performance_metrics()
            
            long_signals = signals['entry_long'].sum() if 'entry_long' in signals.columns else 0
            short_signals = signals['entry_short'].sum() if 'entry_short' in signals.columns else 0
            
            print(f"✅ Results:")
            print(f"   Signals: {long_signals} long, {short_signals} short")
            print(f"   Trades: {len(strategy.closed_positions)}")
            if metrics:
                print(f"   Win rate: {metrics.get('win_rate', 0):.1f}%")
                print(f"   Total PnL: ${metrics.get('total_pnl', 0):.2f}")
                print(f"   Final capital: ${strategy.current_capital:.2f}")
            
            results.append({
                'name': params['name'],
                'signals': long_signals + short_signals,
                'trades': len(strategy.closed_positions),
                'metrics': metrics,
                'final_capital': strategy.current_capital
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results

def summarize_results(stock_results, crypto_results, param_results):
    """Summarize all test results."""
    print("\n" + "="*50)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*50)
    
    # Stock results summary
    if stock_results:
        print(f"\n📈 STOCK RESULTS ({len(stock_results)} assets tested):")
        total_trades = sum(r['closed_trades'] for r in stock_results)
        avg_signals = np.mean([r['signals'] for r in stock_results])
        
        print(f"   Average signals per asset: {avg_signals:.1f}")
        print(f"   Total completed trades: {total_trades}")
        
        profitable_assets = [r for r in stock_results if r['metrics'] and r['metrics'].get('total_pnl', 0) > 0]
        print(f"   Profitable assets: {len(profitable_assets)}/{len(stock_results)}")
        
        if profitable_assets:
            best_stock = max(profitable_assets, key=lambda x: x['metrics']['total_pnl'])
            print(f"   Best performer: {best_stock['symbol']} (${best_stock['metrics']['total_pnl']:.2f} PnL)")
    
    # Crypto results summary
    if crypto_results:
        print(f"\n₿ CRYPTO RESULTS ({len(crypto_results)} assets tested):")
        total_trades = sum(r['closed_trades'] for r in crypto_results)
        avg_signals = np.mean([r['signals'] for r in crypto_results])
        
        print(f"   Average signals per asset: {avg_signals:.1f}")
        print(f"   Total completed trades: {total_trades}")
        
        profitable_assets = [r for r in crypto_results if r['metrics'] and r['metrics'].get('total_pnl', 0) > 0]
        print(f"   Profitable assets: {len(profitable_assets)}/{len(crypto_results)}")
        
        if profitable_assets:
            best_crypto = max(profitable_assets, key=lambda x: x['metrics']['total_pnl'])
            print(f"   Best performer: {best_crypto['symbol']} (${best_crypto['metrics']['total_pnl']:.2f} PnL)")
    
    # Parameter results summary
    if param_results:
        print(f"\n⚙️ PARAMETER TESTING RESULTS:")
        for result in param_results:
            pnl = result['metrics'].get('total_pnl', 0) if result['metrics'] else 0
            print(f"   {result['name']}: {result['trades']} trades, ${pnl:.2f} PnL")
        
        if any(r['metrics'] for r in param_results):
            best_params = max([r for r in param_results if r['metrics']], 
                            key=lambda x: x['metrics']['total_pnl'])
            print(f"   Best parameter set: {best_params['name']}")

def main():
    """Run comprehensive strategy testing."""
    print("🚀 COMPREHENSIVE CTA STRATEGY TESTING")
    print("Testing strategy across multiple assets and parameter sets...")
    
    # Test stocks
    stock_results = test_multiple_stocks()
    
    # Test crypto
    crypto_results = test_multiple_crypto()
    
    # Test different parameters
    param_results = test_different_parameters()
    
    # Summarize everything
    summarize_results(stock_results, crypto_results, param_results)
    
    print(f"\n✅ Testing completed!")
    print(f"   Stock assets tested: {len(stock_results)}")
    print(f"   Crypto assets tested: {len(crypto_results)}")
    print(f"   Parameter sets tested: {len(param_results)}")

if __name__ == "__main__":
    main()
