"""
Stock data fetcher using yfinance.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
from .base_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)


class StockDataFetcher(BaseDataFetcher):
    """Fetches stock data using yfinance."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize stock data fetcher.
        
        Args:
            cache_dir: Directory to cache data files
        """
        super().__init__(cache_dir)
        
    def fetch_data(self, symbol: str, timeframe: str, start_date: str,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'GOOGL')
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframes to yfinance intervals
            interval_map = {
                '1m': '1m',
                '2m': '2m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '90m': '90m',
                '1d': '1d',
                '5d': '5d',
                '1wk': '1wk',
                '1mo': '1mo',
                '3mo': '3mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Determine the period for intraday data
            if interval in ['1m', '2m', '5m', '15m', '30m', '1h', '90m']:
                # For intraday data, yfinance has limitations
                # Calculate days between start and end
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
                days_diff = (end_dt - start_dt).days
                
                # yfinance limits for intraday data
                if days_diff > 60:
                    logger.warning(f"Intraday data limited to 60 days. Adjusting start date.")
                    start_date = (end_dt - timedelta(days=60)).strftime('%Y-%m-%d')
            
            # Fetch data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return pd.DataFrame()
            
            # Select only OHLCV columns
            df = df[required_columns]
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock information.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_stocks_data(self, symbols: List[str], timeframe: str = '1d',
                                start_date: str = '2023-01-01', 
                                end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock tickers
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.get_data(symbol, timeframe, start_date, end_date)
            if not df.empty:
                results[symbol] = df
            else:
                logger.warning(f"No data retrieved for {symbol}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results


# Convenience function
def get_stock_data(symbol: str, timeframe: str = '1d',
                  start_date: str = '2023-01-01', end_date: Optional[str] = None,
                  use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to fetch stock data.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        timeframe: Timeframe (default: '1d')
        start_date: Start date (default: '2023-01-01')
        end_date: End date (optional)
        use_cache: Whether to use cached data (default: True)
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = StockDataFetcher()
    return fetcher.get_data(symbol, timeframe, start_date, end_date, use_cache)


def get_sp500_symbols() -> List[str]:
    """
    Get S&P 500 stock symbols.
    
    Returns:
        List of S&P 500 tickers
    """
    try:
        # Fetch S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        
        # Clean symbols (remove dots, etc.)
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        
        logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to get S&P 500 symbols: {e}")
        # Return a default list of popular stocks
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
