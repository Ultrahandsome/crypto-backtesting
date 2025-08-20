"""
Test data fetching functionality.
"""
import sys
sys.path.append('src')

from data import get_stock_data, get_crypto_data
import pandas as pd
from datetime import datetime, timedelta

def test_stock_data_fetching():
    """Test stock data fetching."""
    print("=== Stock Data Fetching Test ===")
    
    try:
        # Test basic stock data fetching
        data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2023-12-31')
        
        if data.empty:
            print("❌ No stock data returned")
            return False
        
        print(f"✅ Stock data fetched: {len(data)} rows")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Stock data fetching failed: {e}")
        return False

def test_crypto_data_fetching():
    """Test crypto data fetching."""
    print("\n=== Crypto Data Fetching Test ===")
    
    try:
        # Test basic crypto data fetching
        data = get_crypto_data('BTC/USDT', start_date='2023-01-01', end_date='2023-12-31')
        
        if data.empty:
            print("❌ No crypto data returned")
            return False
        
        print(f"✅ Crypto data fetched: {len(data)} rows")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Crypto data fetching failed: {e}")
        return False

def main():
    """Run data fetching tests."""
    print("📊 DATA FETCHING TEST SUITE")
    print("="*40)
    
    test1 = test_stock_data_fetching()
    test2 = test_crypto_data_fetching()
    
    print("\n" + "="*40)
    print("📊 DATA FETCHING TEST SUMMARY")
    print("="*40)
    
    tests = [
        ("Stock Data Fetching", test1),
        ("Crypto Data Fetching", test2)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All data fetching tests passed!")
    else:
        print("⚠️ Some tests failed.")

if __name__ == "__main__":
    main()
