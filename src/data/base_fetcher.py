"""
Base data fetcher class for common functionality.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the base data fetcher.
        
        Args:
            cache_dir: Directory to cache data files
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for a given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'AAPL')
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Forward fill missing values
        df = df.ffill()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure positive prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].abs()
                
        # Ensure high >= low
        if 'high' in df.columns and 'low' in df.columns:
            df['high'] = np.maximum(df['high'], df['low'])
            
        # Ensure volume is positive
        if 'volume' in df.columns:
            df['volume'] = df['volume'].abs()
            
        logger.info(f"Cleaned data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        return df
    
    def save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save DataFrame to cache.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if df.empty:
            return
            
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = self.cache_dir / filename
        
        try:
            df.to_parquet(filepath)
            logger.info(f"Saved {len(df)} rows to cache: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Cached DataFrame or None if not found
        """
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            return None
            
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} rows from cache: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
    
    def get_data(self, symbol: str, timeframe: str, start_date: str,
                 end_date: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get data with caching support.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            OHLCV DataFrame
        """
        # Try to load from cache first
        if use_cache:
            cached_df = self.load_from_cache(symbol, timeframe)
            if cached_df is not None and not cached_df.empty:
                # Check if cached data covers the requested period
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

                # Handle timezone awareness
                if cached_df.index.tz is not None:
                    if start_dt.tz is None:
                        start_dt = start_dt.tz_localize(cached_df.index.tz)
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize(cached_df.index.tz)
                else:
                    if start_dt.tz is not None:
                        start_dt = start_dt.tz_localize(None)
                    if end_dt.tz is not None:
                        end_dt = end_dt.tz_localize(None)

                if (cached_df.index[0] <= start_dt and
                    cached_df.index[-1] >= end_dt):
                    # Filter cached data to requested period
                    mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                    return cached_df[mask]
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {symbol} {timeframe}")
        df = self.fetch_data(symbol, timeframe, start_date, end_date)
        
        if not df.empty:
            df = self.clean_data(df)
            if use_cache:
                self.save_to_cache(df, symbol, timeframe)
        
        return df
