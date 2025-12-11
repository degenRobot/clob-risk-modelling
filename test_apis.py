#!/usr/bin/env python3
"""Test script to verify API connections"""

import sys
sys.path.append('risk-model/src')

from risk_model import binance_data, uniswap_data, config
import json

def test_binance_api():
    """Test Binance API endpoints"""
    print("=== Testing Binance API ===")
    
    try:
        # Test with ETH
        symbol = "ETHUSDT"
        fetcher = binance_data.BinanceDataFetcher()
        
        # Test orderbook
        print(f"\n1. Testing orderbook for {symbol}...")
        orderbook = fetcher.get_orderbook(symbol, limit=10, use_futures=True)
        print(f"✓ Orderbook fetched: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
        print(f"  Best bid: ${orderbook['bids'][0][0]:.2f}")
        print(f"  Best ask: ${orderbook['asks'][0][0]:.2f}")
        
        # Test ticker
        print(f"\n2. Testing 24h ticker...")
        ticker = fetcher.get_ticker_24h(symbol, use_futures=True)
        print(f"✓ Ticker fetched:")
        print(f"  24h Volume: ${float(ticker['quoteVolume']):,.0f}")
        print(f"  24h Change: {float(ticker['priceChangePercent']):.2f}%")
        
        # Test klines
        print(f"\n3. Testing historical klines...")
        klines = fetcher.get_klines(symbol, interval='1h', lookback_days=1, use_futures=True)
        print(f"✓ Klines fetched: {len(klines)} candles")
        print(f"  Latest close: ${klines['close'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_uniswap_api():
    """Test Uniswap API"""
    print("\n=== Testing Uniswap API ===")
    
    try:
        # Test with ETH/USDC pool
        pool_id = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        fetcher = uniswap_data.UniswapDataFetcher()
        
        print(f"\n1. Testing pool query for {pool_id[:10]}...")
        pool = fetcher.query_pool(pool_id)
        
        if pool:
            print(f"✓ Pool data fetched:")
            print(f"  Pair: {pool['token0']['symbol']}/{pool['token1']['symbol']}")
            print(f"  TVL: ${float(pool['totalValueLockedUSD']):,.0f}")
            print(f"  Fee tier: {int(pool['feeTier'])/10000}%")
            return True
        else:
            print("✗ No pool data returned")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: Uniswap API may require a valid subgraph endpoint")
        return False

def test_config():
    """Test configuration loading"""
    print("\n=== Testing Configuration ===")
    
    try:
        markets = config.load_markets()
        print(f"✓ Loaded {len(markets['markets'])} markets")
        
        for market in markets['markets'][:3]:
            print(f"  - {market['name']}: {market['binance_symbol_perp']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("CLOB Risk Model - API Connection Test\n")
    
    # Test each API
    binance_ok = test_binance_api()
    uniswap_ok = test_uniswap_api()
    config_ok = test_config()
    
    # Summary
    print("\n=== Summary ===")
    print(f"Binance API: {'✓ Working' if binance_ok else '✗ Failed'}")
    print(f"Uniswap API: {'✓ Working' if uniswap_ok else '✗ Failed (may need auth)'}")
    print(f"Configuration: {'✓ Loaded' if config_ok else '✗ Failed'}")
    
    if binance_ok and config_ok:
        print("\nCore functionality is working! You can proceed with the notebooks.")
    else:
        print("\nSome APIs failed. Check the errors above.")