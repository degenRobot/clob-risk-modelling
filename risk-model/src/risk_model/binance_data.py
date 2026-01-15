"""Binance data fetching module for CLOB risk modelling"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import time
import logging
import json

logger = logging.getLogger(__name__)

# Constants for kline fetching
SPOT_KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]
KLINE_OUTPUT_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

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

    def get_spot_klines(
        self,
        symbol: str,
        interval: str = "1s",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[List]:
        """
        Fetch spot klines for a single batch (supports 1s interval).

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1s, 1m, etc.)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Max 1000 for spot API

        Returns:
            List of kline data arrays
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._make_request(url, params)

    def fetch_spot_klines_range(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Fetch spot klines for a date range with automatic batching.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1s, 1m, etc.)
            start_dt: Start datetime
            end_dt: End datetime
            progress_callback: Optional callback(current_batch, total_batches)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Calculate interval in ms
        interval_ms = self._interval_to_ms(interval)
        total_points = (end_ms - start_ms) // interval_ms
        total_batches = (total_points // 1000) + 1  # Spot API max is 1000

        all_data = []
        current_start = start_ms
        batch_num = 0

        while current_start < end_ms:
            batch_num += 1
            if progress_callback:
                progress_callback(batch_num, total_batches)

            try:
                data = self.get_spot_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_ms,
                    limit=1000
                )

                if not data:
                    logger.warning(f"No data returned for batch {batch_num}")
                    break

                all_data.extend(data)

                # Move to next batch
                last_timestamp = data[-1][0]
                current_start = last_timestamp + interval_ms

            except Exception as e:
                logger.error(f"Error fetching batch {batch_num}: {e}")
                raise

        return self._convert_spot_klines_to_df(all_data)

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        unit = interval[-1]
        value = int(interval[:-1]) if len(interval) > 1 else 1

        multipliers = {
            's': 1000,
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }
        return value * multipliers.get(unit, 1000)

    def _convert_spot_klines_to_df(self, data: List[List]) -> pd.DataFrame:
        """Convert raw spot kline data to DataFrame"""
        if not data:
            return pd.DataFrame(columns=KLINE_OUTPUT_COLUMNS)

        df = pd.DataFrame(data, columns=SPOT_KLINE_COLUMNS)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[KLINE_OUTPUT_COLUMNS].copy()


class SpotKlineDownloader:
    """
    Downloads large amounts of spot kline data with:
    - Daily batching for reliability
    - Progress tracking
    - Resume capability
    - Intermediate saves

    Supports 1s interval for high-frequency analysis.
    """

    def __init__(
        self,
        output_dir: str = "data/spot_klines",
        rate_limit_delay: float = 0.05
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = BinanceDataFetcher()
        self.fetcher.rate_limit_delay = rate_limit_delay

    def download_range(
        self,
        symbol: str,
        interval: str = "1s",
        end_date: Optional[datetime] = None,
        days: int = 365,
        save_daily: bool = True
    ) -> pd.DataFrame:
        """
        Download kline data for a date range.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1s, 1m, etc.)
            end_date: End date (defaults to now)
            days: Number of days to fetch
            save_daily: Whether to save daily CSV files

        Returns:
            Combined DataFrame of all data
        """
        if end_date is None:
            end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        start_date = end_date - timedelta(days=days)

        progress_file = self.output_dir / f"{symbol}_{interval}_progress.json"
        completed_days = self._load_progress(progress_file)

        all_dfs = []
        current_date = start_date

        print(f"Downloading {symbol} {interval} spot data from {start_date.date()} to {end_date.date()}")
        print(f"Already completed: {len(completed_days)} days")

        # Estimate time for 1s data
        if interval == "1s":
            remaining_days = days - len(completed_days)
            est_requests = remaining_days * 87  # ~86,400 seconds/day / 1000 per request
            est_minutes = est_requests * 0.05 / 60  # rate_limit_delay
            print(f"Estimated time for remaining {remaining_days} days: ~{est_minutes:.0f} minutes")

        while current_date < end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_file = self.output_dir / f"{symbol}_{interval}_{date_str}.csv"

            # Check if already downloaded
            if date_str in completed_days and daily_file.exists():
                print(f"  {date_str}: Loading from cache")
                df = pd.read_csv(daily_file, parse_dates=['timestamp'])
                all_dfs.append(df)
                current_date += timedelta(days=1)
                continue

            # Fetch this day's data
            day_start = current_date
            day_end = current_date + timedelta(days=1)

            print(f"  {date_str}: Fetching...", end=" ", flush=True)

            try:
                df = self.fetcher.fetch_spot_klines_range(
                    symbol=symbol,
                    interval=interval,
                    start_dt=day_start,
                    end_dt=day_end
                )

                print(f"got {len(df)} rows")

                if save_daily and len(df) > 0:
                    df.to_csv(daily_file, index=False)
                    completed_days.add(date_str)
                    self._save_progress(progress_file, completed_days)

                all_dfs.append(df)

            except Exception as e:
                print(f"ERROR: {e}")
                logger.error(f"Failed to fetch {date_str}: {e}")
                # Continue with next day rather than failing entirely

            current_date += timedelta(days=1)

        # Combine all data
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

            # Save combined file
            combined_file = self.output_dir / f"{symbol}_{interval}_combined.csv"
            combined.to_csv(combined_file, index=False)
            print(f"\nSaved combined data to {combined_file}")
            print(f"Total rows: {len(combined)}")

            return combined

        return pd.DataFrame(columns=KLINE_OUTPUT_COLUMNS)

    def _load_progress(self, progress_file: Path) -> set:
        """Load set of completed dates from progress file"""
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_progress(self, progress_file: Path, completed: set):
        """Save completed dates to progress file"""
        with open(progress_file, 'w') as f:
            json.dump(list(completed), f)


# Backwards compatibility alias
MarkPriceKlineDownloader = SpotKlineDownloader


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


def download_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1s",
    days: int = 365,
    output_dir: str = "data/spot_klines"
) -> pd.DataFrame:
    """
    Convenience function to download spot klines (supports 1s interval).

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1s', '1m')
        days: Number of days to fetch
        output_dir: Directory to save CSV files

    Returns:
        Combined DataFrame of all data
    """
    downloader = SpotKlineDownloader(output_dir=output_dir)
    return downloader.download_range(symbol=symbol, interval=interval, days=days)


# Backwards compatibility alias
download_mark_price_klines = download_spot_klines