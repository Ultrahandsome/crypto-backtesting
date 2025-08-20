"""
Test script to verify technical indicators functionality.
"""
import sys
sys.path.append('src')

from data import get_stock_data
from indicators import sma, ema, rsi, macd, ma_crossover
import pandas as pd

def test_indicators():
    """Test technical indicators."""
    print("=== Technical Indicators Test ===")
    
    # Get some test data
    print("Fetching test data...")
    data = get_stock_data('AAPL', timeframe='1d', start_date='2024-01-01', end_date='2024-03-01')
    
    if data.empty:
        print("❌ No data available for testing")
        return False
    
    print(f"✅ Test data: {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Test SMA
    print("\nTesting Simple Moving Average...")
    try:
        sma_20 = sma(data, period=20)
        print(f"✅ SMA(20): {len(sma_20)} values, last value: {sma_20.iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ SMA failed: {e}")
        return False
    
    # Test EMA
    print("\nTesting Exponential Moving Average...")
    try:
        ema_20 = ema(data, period=20)
        print(f"✅ EMA(20): {len(ema_20)} values, last value: {ema_20.iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ EMA failed: {e}")
        return False
    
    # Test RSI
    print("\nTesting RSI...")
    try:
        rsi_14 = rsi(data, period=14)
        print(f"✅ RSI(14): {len(rsi_14)} values, last value: {rsi_14.iloc[-1]:.2f}")
        print(f"   RSI range: {rsi_14.min():.2f} - {rsi_14.max():.2f}")
    except Exception as e:
        print(f"❌ RSI failed: {e}")
        return False
    
    # Test MACD
    print("\nTesting MACD...")
    try:
        macd_data = macd(data, fast=12, slow=26, signal=9)
        print(f"✅ MACD: {len(macd_data)} rows")
        print(f"   Columns: {list(macd_data.columns)}")
        print(f"   Last MACD: {macd_data['MACD'].iloc[-1]:.4f}")
        print(f"   Last Signal: {macd_data['Signal'].iloc[-1]:.4f}")
    except Exception as e:
        print(f"❌ MACD failed: {e}")
        return False
    
    # Test MA Crossover
    print("\nTesting MA Crossover...")
    try:
        crossover_data = ma_crossover(data, fast_period=10, slow_period=30, ma_type='sma')
        print(f"✅ MA Crossover: {len(crossover_data)} rows")
        print(f"   Columns: {list(crossover_data.columns)}")
        bullish_signals = crossover_data['bullish_crossover'].sum()
        bearish_signals = crossover_data['bearish_crossover'].sum()
        print(f"   Bullish crossovers: {bullish_signals}")
        print(f"   Bearish crossovers: {bearish_signals}")
    except Exception as e:
        print(f"❌ MA Crossover failed: {e}")
        return False
    
    # Test indicator combination (CTA strategy preview)
    print("\nTesting CTA Strategy Indicators Combination...")
    try:
        # Calculate all indicators needed for CTA strategy
        fast_ma = sma(data, period=10)
        slow_ma = sma(data, period=30)
        rsi_values = rsi(data, period=14)
        
        # Create a simple signal
        signals = pd.DataFrame(index=data.index)
        signals['fast_ma'] = fast_ma
        signals['slow_ma'] = slow_ma
        signals['rsi'] = rsi_values
        signals['ma_signal'] = (fast_ma > slow_ma).astype(int)
        signals['rsi_signal'] = ((rsi_values > 30) & (rsi_values < 70)).astype(int)
        signals['combined_signal'] = signals['ma_signal'] & signals['rsi_signal']
        
        buy_signals = signals['combined_signal'].sum()
        print(f"✅ CTA Strategy Preview:")
        print(f"   Total periods: {len(signals)}")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Signal rate: {buy_signals/len(signals)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ CTA Strategy combination failed: {e}")
        return False
    
    print("\n🎉 All indicator tests passed!")
    return True

if __name__ == "__main__":
    success = test_indicators()
    if not success:
        print("⚠️  Some indicator tests failed.")
        sys.exit(1)
