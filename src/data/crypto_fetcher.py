"""
Cryptocurrency data fetcher using CCXT.
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time
import logging
from .base_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)


class CryptoDataFetcher(BaseDataFetcher):
    """Fetches cryptocurrency data using CCXT."""

    def __init__(self, exchange_name: str = 'coinbase', cache_dir: Optional[str] = None):
        """
        Initialize crypto data fetcher.
        
        Args:
            exchange_name: Name of the exchange (default: binance)
            cache_dir: Directory to cache data files
        """
        super().__init__(cache_dir)
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize the exchange connection."""
        # Try multiple exchanges in order of preference
        exchanges_to_try = [self.exchange_name, 'coinbase', 'kraken', 'bitfinex']

        for exchange_name in exchanges_to_try:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': '',  # Not needed for public data
                    'secret': '',
                    'timeout': 30000,
                    'enableRateLimit': True,
                    'sandbox': False,
                })

                # Test connection
                exchange.load_markets()
                logger.info(f"Connected to {exchange_name} exchange")
                self.exchange_name = exchange_name  # Update to working exchange
                return exchange

            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_name}: {e}")
                continue

        # If all exchanges fail, raise the last error
        raise Exception("Failed to connect to any cryptocurrency exchange")
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from the exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else None
            
            # Fetch data in chunks to avoid rate limits
            all_data = []
            current_ts = start_ts
            limit = 1000  # Max candles per request
            
            while True:
                try:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_ts,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    
                    # Update timestamp for next batch
                    current_ts = ohlcv[-1][0] + 1
                    
                    # Check if we've reached the end date
                    if end_ts and current_ts >= end_ts:
                        break
                        
                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except ccxt.BaseError as e:
                    logger.error(f"Error fetching data: {e}")
                    break
            
            if not all_data:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end date if specified
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> Dict[str, Any]:
        """Get available trading symbols from the exchange."""
        try:
            markets = self.exchange.load_markets()
            symbols = {}
            
            for symbol, market in markets.items():
                if market['active'] and market['type'] == 'spot':
                    symbols[symbol] = {
                        'base': market['base'],
                        'quote': market['quote'],
                        'active': market['active'],
                        'min_amount': market.get('limits', {}).get('amount', {}).get('min'),
                        'max_amount': market.get('limits', {}).get('amount', {}).get('max'),
                    }
            
            logger.info(f"Found {len(symbols)} active symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return {}
    
    def get_available_timeframes(self) -> list:
        """Get available timeframes from the exchange."""
        try:
            timeframes = list(self.exchange.timeframes.keys())
            logger.info(f"Available timeframes: {timeframes}")
            return timeframes
        except Exception as e:
            logger.error(f"Failed to get timeframes: {e}")
            return ['1m', '5m', '15m', '1h', '4h', '1d']


# Convenience function
def get_crypto_data(symbol: str, timeframe: str = '1h',
                   start_date: str = '2023-01-01', end_date: Optional[str] = None,
                   exchange: str = 'coinbase', use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to fetch crypto data.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (default: '1h')
        start_date: Start date (default: '2023-01-01')
        end_date: End date (optional)
        exchange: Exchange name (default: 'binance')
        use_cache: Whether to use cached data (default: True)
        
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = CryptoDataFetcher(exchange_name=exchange)
    return fetcher.get_data(symbol, timeframe, start_date, end_date, use_cache)
