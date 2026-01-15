#!/usr/bin/env python3
"""
Memory-efficient kline downloader - appends to CSV after each API call.

Features:
- Constant memory usage (doesn't accumulate data in memory)
- Resume capability (reads last timestamp from CSV)
- Append-only writes (no need to reload entire file)

Usage:
    poetry run python scripts/download_klines.py --symbol BTCUSDT --days 7
    poetry run python scripts/download_klines.py --symbol BTCUSDT --days 365
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests


class KlineDownloader:
    """Memory-efficient kline downloader - appends to CSV after each API call."""

    BASE_URL = "https://api.binance.com"
    COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        output_dir: str = "data/spot_klines",
        rate_limit_delay: float = 0.05,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_output_path(self, symbol: str, interval: str = "1s") -> Path:
        return self.output_dir / f"{symbol}_{interval}.csv"

    def get_last_timestamp(self, csv_path: Path) -> Optional[datetime]:
        """Read only the last line of CSV to get last timestamp."""
        if not csv_path.exists():
            return None
        try:
            # Read last line efficiently (seek from end)
            with open(csv_path, "rb") as f:
                f.seek(-2, 2)  # Go to 2nd last byte
                while f.read(1) != b"\n":
                    f.seek(-2, 1)
                last_line = f.readline().decode().strip()

            if last_line and not last_line.startswith("timestamp"):
                ts_str = last_line.split(",")[0]
                return datetime.fromisoformat(ts_str)
        except Exception as e:
            print(f"  Warning: Could not read last timestamp: {e}")
        return None

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1000,
    ) -> list:
        """Fetch a single batch of klines."""
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        time.sleep(self.rate_limit_delay)
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def download(
        self,
        symbol: str,
        interval: str = "1s",
        days: int = 365,
        end_date: Optional[datetime] = None,
    ):
        """Download klines with append-to-CSV strategy (constant memory)."""
        if end_date is None:
            end_date = datetime.utcnow()

        target_start = end_date - timedelta(days=days)
        csv_path = self.get_output_path(symbol, interval)

        print(f"\n{'='*60}", flush=True)
        print(f"Downloading {symbol} {interval}", flush=True)
        print(f"  Target: {target_start.date()} to {end_date.date()}", flush=True)
        print(f"  Output: {csv_path}", flush=True)
        print(f"{'='*60}", flush=True)

        # Check for resume
        last_ts = self.get_last_timestamp(csv_path)
        if last_ts:
            actual_start = last_ts + timedelta(seconds=1)
            print(f"  Resuming from: {last_ts}", flush=True)
            file_mode = "a"  # Append
            write_header = False
        else:
            actual_start = target_start
            print(f"  Starting fresh", flush=True)
            file_mode = "w"  # Write new
            write_header = True

        if actual_start >= end_date:
            print(f"  Already up to date!", flush=True)
            return

        # Calculate estimates
        interval_ms = self._interval_to_ms(interval)
        total_seconds = (end_date - actual_start).total_seconds()
        total_points = int(total_seconds * 1000 / interval_ms)
        total_requests = (total_points // 1000) + 1

        print(f"  Est. API calls: {total_requests:,}", flush=True)
        print(f"  Est. time: {total_requests * self.rate_limit_delay / 60:.1f} min", flush=True)
        print(flush=True)

        # Open CSV for writing/appending
        start_ms = int(actual_start.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        request_count = 0
        total_rows = 0
        start_time = time.time()

        with open(csv_path, file_mode, newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self.COLUMNS)

            current_start = start_ms

            while current_start < end_ms:
                request_count += 1

                try:
                    data = self.fetch_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=end_ms,
                        limit=1000,
                    )

                    if not data:
                        print(f"\n  No more data available", flush=True)
                        break

                    # Write rows immediately to CSV
                    for row in data:
                        ts = datetime.utcfromtimestamp(row[0] / 1000).isoformat(sep=" ")
                        writer.writerow([
                            ts,
                            float(row[1]),  # open
                            float(row[2]),  # high
                            float(row[3]),  # low
                            float(row[4]),  # close
                            float(row[5]),  # volume
                        ])

                    total_rows += len(data)
                    f.flush()  # Flush to disk

                    # Progress
                    last_timestamp = data[-1][0]
                    current_start = last_timestamp + interval_ms

                    if request_count % 100 == 0:
                        progress = (current_start - start_ms) / (end_ms - start_ms) * 100
                        elapsed = time.time() - start_time
                        rate = request_count / elapsed if elapsed > 0 else 0
                        eta = (total_requests - request_count) / rate / 60 if rate > 0 else 0
                        current_dt = datetime.utcfromtimestamp(last_timestamp / 1000)

                        print(f"  [{progress:5.1f}%] {request_count:,} req | "
                              f"{total_rows:,} rows | {current_dt.date()} | "
                              f"ETA: {eta:.1f}m", flush=True)

                except KeyboardInterrupt:
                    print(f"\n  Interrupted! Progress saved.", flush=True)
                    raise
                except Exception as e:
                    print(f"\n  Error: {e}", flush=True)
                    print(f"  Progress saved. Run again to resume.", flush=True)
                    break

        print(f"\n  Done! Total rows written: {total_rows:,}", flush=True)

    def _interval_to_ms(self, interval: str) -> int:
        unit = interval[-1]
        value = int(interval[:-1]) if len(interval) > 1 else 1
        multipliers = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
        return value * multipliers.get(unit, 1000)


def main():
    parser = argparse.ArgumentParser(description="Download Binance klines (memory-efficient)")
    parser.add_argument("--symbol", "-s", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--days", "-d", type=int, default=7)
    parser.add_argument("--interval", "-i", default="1s")
    parser.add_argument("--output-dir", "-o", default="data/spot_klines")
    parser.add_argument("--rate-limit", "-r", type=float, default=0.05)

    args = parser.parse_args()

    downloader = KlineDownloader(
        output_dir=args.output_dir,
        rate_limit_delay=args.rate_limit,
    )

    print(f"Binance Kline Downloader (Memory-Efficient)", flush=True)
    print(f"Symbols: {', '.join(args.symbol)} | Days: {args.days}", flush=True)

    for symbol in args.symbol:
        try:
            downloader.download(symbol=symbol, interval=args.interval, days=args.days)
        except KeyboardInterrupt:
            print(f"\nStopped. Run again to resume.", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
