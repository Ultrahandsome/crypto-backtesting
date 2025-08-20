"""
Test strategy development functionality.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from strategies import CTAStrategy
import pandas as pd
import numpy as np

def test_cta_strategy_creation():
    """Test CTA strategy creation."""
    print("=== CTA Strategy Creation Test ===")
    
    try:
        # Test basic strategy creation
        strategy = CTAStrategy(
            fast_ma_period=10,
            slow_ma_period=30,
            rsi_period=14,
            stop_loss_pct=0.02,
            take_profit_pct=0.06
        )
        
        print("✅ CTA strategy created successfully")
        print(f"   Fast MA: {strategy.fast_ma_period}")
        print(f"   Slow MA: {strategy.slow_ma_period}")
        print(f"   RSI: {strategy.rsi_period}")
        
        return True
        
    except Exception as e:
        print(f"❌ CTA strategy creation failed: {e}")
        return False

def test_signal_generation():
    """Test signal generation."""
    print("\n=== Signal Generation Test ===")
    
    try:
        # Get test data
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2023-12-31')
        
        if data.empty:
            print("❌ No test data available")
            return False
        
        # Create strategy
        strategy = CTAStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(data, 'AAPL')
        
        print(f"✅ Signals generated: {len(signals)} rows")
        print(f"   Columns: {list(signals.columns)}")
        
        # Check for signal columns
        expected_columns = ['fast_ma', 'slow_ma', 'rsi', 'signal']
        missing_columns = [col for col in expected_columns if col not in signals.columns]
        
        if missing_columns:
            print(f"⚠️ Missing signal columns: {missing_columns}")
        else:
            print("✅ All expected signal columns present")
        
        # Count signals
        if 'signal' in signals.columns:
            buy_signals = (signals['signal'] > 0).sum()
            sell_signals = (signals['signal'] < 0).sum()
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def test_strategy_parameters():
    """Test strategy parameter validation."""
    print("\n=== Strategy Parameters Test ===")
    
    try:
        # Test valid parameters
        strategy1 = CTAStrategy(fast_ma_period=5, slow_ma_period=20)
        print("✅ Valid parameters accepted")
        
        # Test edge cases
        strategy2 = CTAStrategy(fast_ma_period=1, slow_ma_period=2)
        print("✅ Edge case parameters handled")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        return False

def main():
    """Run strategy tests."""
    print("🎯 STRATEGY DEVELOPMENT TEST SUITE")
    print("="*45)
    
    test1 = test_cta_strategy_creation()
    test2 = test_signal_generation()
    test3 = test_strategy_parameters()
    
    print("\n" + "="*45)
    print("🎯 STRATEGY TEST SUMMARY")
    print("="*45)
    
    tests = [
        ("CTA Strategy Creation", test1),
        ("Signal Generation", test2),
        ("Strategy Parameters", test3)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All strategy tests passed!")
    else:
        print("⚠️ Some tests failed.")

if __name__ == "__main__":
    main()
