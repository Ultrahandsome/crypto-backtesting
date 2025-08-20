"""
Test script to verify data fetching functionality.
"""
import sys
sys.path.append('src')

from data import get_stock_data, get_crypto_data
import pandas as pd

def test_stock_data():
    """Test stock data fetching."""
    print("Testing stock data fetching...")
    try:
        # Fetch Apple stock data
        df = get_stock_data('AAPL', timeframe='1d', start_date='2024-01-01', end_date='2024-02-01')
        print(f"✅ Stock data: {len(df)} rows fetched for AAPL")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        return True
    except Exception as e:
        print(f"❌ Stock data fetch failed: {e}")
        return False

def test_crypto_data():
    """Test crypto data fetching."""
    print("\nTesting crypto data fetching...")
    try:
        # Fetch Bitcoin data
        df = get_crypto_data('BTC/USDT', timeframe='1d', start_date='2024-01-01', end_date='2024-02-01')
        print(f"✅ Crypto data: {len(df)} rows fetched for BTC/USDT")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        return True
    except Exception as e:
        print(f"❌ Crypto data fetch failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Data Fetching Test ===")
    
    stock_success = test_stock_data()
    crypto_success = test_crypto_data()
    
    print(f"\n=== Results ===")
    print(f"Stock data: {'✅ PASS' if stock_success else '❌ FAIL'}")
    print(f"Crypto data: {'✅ PASS' if crypto_success else '❌ FAIL'}")
    
    if stock_success and crypto_success:
        print("🎉 All data fetching tests passed!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
