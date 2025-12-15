"""Binance data fetching module for CLOB risk modelling"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    """Fetches market data from Binance REST API"""
    
    def __init__(self, base_url: str = "https://api.binance.com", 
                 futures_url: str = "https://fapi.binance.com"):
        self.base_url = base_url
        self.futures_url = futures_url
        self.session = requests.Session()
        self.rate_limit_delay = 0.05  # 50ms between requests
        
    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """Make HTTP request with rate limiting and error handling"""
        time.sleep(self.rate_limit_delay)
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
            
    def get_orderbook(self, symbol: str, limit: int = 500, use_futures: bool = True) -> dict:
        """
        Fetch orderbook snapshot for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            limit: Number of price levels (5, 10, 20, 50, 100, 500, 1000, 5000)
            use_futures: Whether to use futures API endpoint
            
        Returns:
            Dictionary with 'bids' and 'asks' arrays
        """
        base = self.futures_url if use_futures else self.base_url
        endpoint = "/fapi/v1/depth" if use_futures else "/api/v3/depth"
        url = f"{base}{endpoint}"
        
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        data = self._make_request(url, params)
        
        # Convert to more usable format with error handling
        if "bids" not in data or "asks" not in data:
            logger.error(f"Invalid orderbook response for {symbol}: missing bids/asks")
            return {
                "bids": np.array([]),
                "asks": np.array([]),
                "lastUpdateId": data.get("lastUpdateId", 0),
                "timestamp": datetime.now()
            }
            
        return {
            "bids": np.array(data["bids"], dtype=float),
            "asks": np.array(data["asks"], dtype=float),
            "lastUpdateId": data["lastUpdateId"],
            "timestamp": datetime.now()
        }
        
    def get_klines(self, symbol: str, interval: str = "1h", 
                   lookback_days: int = 30, use_futures: bool = True) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            lookback_days: Number of days to look back
            use_futures: Whether to use futures API
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        base = self.futures_url if use_futures else self.base_url
        endpoint = "/fapi/v1/klines" if use_futures else "/api/v3/klines"
        url = f"{base}{endpoint}"
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Max allowed
        }
        
        all_klines = []
        
        while start_time < end_time:
            data = self._make_request(url, params)
            if not data:
                break
                
            all_klines.extend(data)
            
            # Update start time for next batch
            if data:
                start_time = data[-1][0] + 1
                params["startTime"] = start_time
            else:
                break
                
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
            
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def get_ticker_24h(self, symbol: str, use_futures: bool = True) -> dict:
        """
        Get 24hr ticker statistics
        
        Returns:
            Dictionary with price change stats and volume
        """
        base = self.futures_url if use_futures else self.base_url
        endpoint = "/fapi/v1/ticker/24hr" if use_futures else "/api/v3/ticker/24hr"
        url = f"{base}{endpoint}"
        
        params = {"symbol": symbol}
        
        return self._make_request(url, params)
        
    def get_book_ticker(self, symbol: str, use_futures: bool = True) -> dict:
        """
        Get best bid/ask prices and quantities
        
        Returns:
            Dictionary with best bid/ask prices and sizes
        """
        base = self.futures_url if use_futures else self.base_url
        endpoint = "/fapi/v1/ticker/bookTicker" if use_futures else "/api/v3/ticker/bookTicker"
        url = f"{base}{endpoint}"
        
        params = {"symbol": symbol}
        
        return self._make_request(url, params)
        
    def get_exchange_info(self, use_futures: bool = True) -> dict:
        """Get exchange trading rules and symbol information"""
        base = self.futures_url if use_futures else self.base_url
        endpoint = "/fapi/v1/exchangeInfo" if use_futures else "/api/v3/exchangeInfo"
        url = f"{base}{endpoint}"
        
        return self._make_request(url)

# Convenience functions
def get_orderbook(symbol: str, limit: int = 500, use_futures: bool = True) -> dict:
    """Convenience function to fetch orderbook"""
    fetcher = BinanceDataFetcher()
    return fetcher.get_orderbook(symbol, limit, use_futures)

def get_klines(symbol: str, interval: str = "1h", 
               lookback_days: int = 30, use_futures: bool = True) -> pd.DataFrame:
    """Convenience function to fetch klines"""
    fetcher = BinanceDataFetcher()
    return fetcher.get_klines(symbol, interval, lookback_days, use_futures)

def fetch_all_data(symbols: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Fetch all relevant data for given symbols
    
    Args:
        symbols: List of symbols to fetch. If None, uses default list
        
    Returns:
        Dictionary with market data for each symbol
    """
    if symbols is None:
        symbols = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
        
    fetcher = BinanceDataFetcher()
    results = {}
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")
        try:
            results[symbol] = {
                "orderbook": fetcher.get_orderbook(symbol),
                "ticker_24h": fetcher.get_ticker_24h(symbol),
                "book_ticker": fetcher.get_book_ticker(symbol),
                "klines_1h": fetcher.get_klines(symbol, "1h", 30),
                "klines_1d": fetcher.get_klines(symbol, "1d", 90)
            }
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            results[symbol] = {"error": str(e)}
            
    return results