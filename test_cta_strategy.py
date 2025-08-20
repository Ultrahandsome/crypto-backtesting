"""
Test script to verify CTA strategy functionality.
"""
import sys
sys.path.append('src')

from data import get_stock_data, get_crypto_data
from strategies import CTAStrategy
import pandas as pd

def test_cta_strategy():
    """Test CTA strategy implementation."""
    print("=== CTA Strategy Test ===")
    
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
    
    print(f"✅ Strategy initialized: {strategy.name}")
    print(f"   Initial capital: ${strategy.initial_capital:,.2f}")
    print(f"   Parameters: {strategy.parameters}")
    
    # Get test data
    print("\nFetching test data...")
    data = get_stock_data('AAPL', timeframe='1d', start_date='2024-01-01', end_date='2024-03-01')
    
    if data.empty:
        print("❌ No data available for testing")
        return False
    
    print(f"✅ Test data: {len(data)} rows")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Generate signals
    print("\nGenerating signals...")
    try:
        signals = strategy.generate_signals(data, 'AAPL')
        print(f"✅ Signals generated: {len(signals)} rows")
        print(f"   Columns: {list(signals.columns)}")
        
        # Count signals
        long_entries = signals['entry_long'].sum() if 'entry_long' in signals.columns else 0
        short_entries = signals['entry_short'].sum() if 'entry_short' in signals.columns else 0
        exit_signals = signals['exit_signal'].sum() if 'exit_signal' in signals.columns else 0
        
        print(f"   Long entries: {long_entries}")
        print(f"   Short entries: {short_entries}")
        print(f"   Exit signals: {exit_signals}")
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False
    
    # Test position sizing
    print("\nTesting position sizing...")
    try:
        test_price = data['close'].iloc[-1]
        position_size = strategy.calculate_position_size('AAPL', test_price, signal_strength=1.0)
        position_value = position_size * test_price
        
        print(f"✅ Position sizing:")
        print(f"   Test price: ${test_price:.2f}")
        print(f"   Position size: {position_size:.4f} shares")
        print(f"   Position value: ${position_value:.2f}")
        print(f"   % of capital: {position_value/strategy.current_capital*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Position sizing failed: {e}")
        return False
    
    # Test stop loss and take profit calculation
    print("\nTesting stop loss and take profit...")
    try:
        entry_price = 180.0
        sl_long, tp_long = strategy.calculate_stop_loss_take_profit(entry_price, 'long')
        sl_short, tp_short = strategy.calculate_stop_loss_take_profit(entry_price, 'short')
        
        print(f"✅ Risk management levels:")
        print(f"   Entry price: ${entry_price:.2f}")
        print(f"   Long - SL: ${sl_long:.2f}, TP: ${tp_long:.2f}")
        print(f"   Short - SL: ${sl_short:.2f}, TP: ${tp_short:.2f}")
        
    except Exception as e:
        print(f"❌ Risk management calculation failed: {e}")
        return False
    
    # Test signal processing (simulation)
    print("\nTesting signal processing...")
    try:
        # Add signals to data for processing
        data_with_signals = data.copy()
        signals_subset = signals[['entry_long', 'entry_short', 'exit_signal', 'signal_strength']].fillna(False)
        data_with_signals = pd.concat([data_with_signals, signals_subset], axis=1)
        
        # Process first few signals to test
        test_data = data_with_signals.head(20)  # Test with first 20 rows
        actions = strategy.process_signals(test_data, 'AAPL')
        
        print(f"✅ Signal processing:")
        print(f"   Actions generated: {len(actions)}")
        
        if actions:
            print("   Sample actions:")
            for i, action in enumerate(actions[:3]):  # Show first 3 actions
                print(f"     {i+1}. {action['timestamp'].strftime('%Y-%m-%d')}: "
                      f"{action['action']} {action.get('size', 0):.4f} @ ${action['price']:.2f}")
        
        # Check strategy state
        current_positions = strategy.get_current_positions()
        closed_positions = len(strategy.closed_positions)
        
        print(f"   Current positions: {len(current_positions)}")
        print(f"   Closed positions: {closed_positions}")
        print(f"   Current capital: ${strategy.current_capital:.2f}")
        
    except Exception as e:
        print(f"❌ Signal processing failed: {e}")
        return False
    
    # Test performance metrics
    print("\nTesting performance metrics...")
    try:
        metrics = strategy.get_performance_metrics()
        print(f"✅ Performance metrics:")
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("   No closed positions yet")
        
    except Exception as e:
        print(f"❌ Performance metrics failed: {e}")
        return False
    
    print("\n🎉 All CTA strategy tests passed!")
    return True

def test_crypto_cta():
    """Test CTA strategy with crypto data."""
    print("\n=== CTA Strategy Crypto Test ===")
    
    try:
        # Get crypto data
        crypto_data = get_crypto_data('BTC/USDT', timeframe='1d', start_date='2024-01-01', end_date='2024-02-15')
        
        if crypto_data.empty:
            print("❌ No crypto data available")
            return False
        
        print(f"✅ Crypto data: {len(crypto_data)} rows")
        
        # Initialize strategy for crypto
        crypto_strategy = CTAStrategy(
            fast_ma_period=5,
            slow_ma_period=20,
            rsi_period=14,
            position_size_pct=0.05,  # Smaller position for crypto volatility
            initial_capital=50000
        )
        
        # Generate signals
        signals = crypto_strategy.generate_signals(crypto_data, 'BTC/USDT')
        
        long_entries = signals['entry_long'].sum() if 'entry_long' in signals.columns else 0
        short_entries = signals['entry_short'].sum() if 'entry_short' in signals.columns else 0
        
        print(f"✅ Crypto signals: {long_entries} long, {short_entries} short")
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto CTA test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_cta_strategy()
    success2 = test_crypto_cta()
    
    if success1 and success2:
        print("\n🎉 All CTA strategy tests passed!")
    else:
        print("\n⚠️  Some CTA strategy tests failed.")
        sys.exit(1)
