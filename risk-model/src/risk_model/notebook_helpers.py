"""Helper functions for Jupyter notebooks to reduce code duplication"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from . import config, binance_data, pricing, metrics, scenarios, volume_analysis

logger = logging.getLogger(__name__)

def setup_pandas_display():
    """Configure pandas display options for notebooks"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_rows', 100)

def fetch_market_data(markets: List[Dict], lookback_days: int = 30, 
                     limit_markets: Optional[int] = None) -> Dict[str, Dict]:
    """
    Fetch market data for multiple assets
    
    Args:
        markets: List of market configurations
        lookback_days: Days of historical data to fetch
        limit_markets: Limit number of markets to fetch (None = all)
        
    Returns:
        Dictionary of market data by symbol name
    """
    binance = binance_data.BinanceDataFetcher()
    market_data = {}
    
    markets_to_fetch = markets[:limit_markets] if limit_markets else markets
    
    for market in markets_to_fetch:
        symbol = market['binance_symbol_perp']
        
        try:
            # Fetch all data
            orderbook = binance.get_orderbook(symbol, limit=500)
            ticker = binance.get_ticker_24h(symbol)
            klines = binance.get_klines(symbol, interval='1h', lookback_days=lookback_days)
            book_ticker = binance.get_book_ticker(symbol)
            
            # Process liquidity
            liq_curve = pricing.compute_orderbook_liquidity_curve(orderbook)
            cex_depth = pricing.depth_at_impact(liq_curve, 1.0)
            
            # Calculate volatility
            vol = pricing.realized_vol(klines)
            
            # Volume analysis
            vol_analysis, vol_profile = volume_analysis.analyze_volume_patterns(klines)
            
            # Spread metrics
            spread_metrics = pricing.calculate_spread_metrics(book_ticker)
            
            market_data[market['name']] = {
                'symbol': symbol,
                'orderbook': orderbook,
                'liquidity_curve': liq_curve,
                'ticker': ticker,
                'klines': klines,
                'book_ticker': book_ticker,
                'volume_analysis': vol_analysis,
                'volume_profile': vol_profile,
                'current_price': klines['close'].iloc[-1],
                'volume_24h_usd': float(ticker['quoteVolume']),
                'cex_liquidity_usd': cex_depth,
                'total_liquidity_usd': cex_depth,
                'realized_vol': vol,
                'spread_bps': spread_metrics['spread_bps'],
                'spread_metrics': spread_metrics,
                'asset': market['name'].split('-')[0],
                'oracle_meta': {'type': market.get('oracle', 'unknown')},
                'market_config': market
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {market['name']}: {e}")
            continue
            
    return market_data

def get_market_summary(market_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create summary DataFrame from market data
    
    Args:
        market_data: Dictionary of market data
        
    Returns:
        Summary DataFrame
    """
    summary_rows = []
    
    for name, data in market_data.items():
        summary_rows.append({
            'Market': name,
            'Price': data['current_price'],
            '24h Volume': data['volume_24h_usd'],
            'Liquidity @ 1%': data['cex_liquidity_usd'],
            'Volatility (%)': data['realized_vol'] * 100,
            'Spread (bps)': data['spread_bps']
        })
    
    return pd.DataFrame(summary_rows)

def calculate_risk_metrics_all(market_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Calculate risk metrics for all markets
    
    Args:
        market_data: Dictionary of market data
        
    Returns:
        DataFrame with risk metrics
    """
    risk_rows = []
    
    for name, data in market_data.items():
        # Calculate all metrics
        all_metrics = metrics.calculate_all_metrics(data)
        
        risk_rows.append({
            'Market': name,
            'Liquidity Score': all_metrics['scores']['liquidity'],
            'Volatility Score': all_metrics['scores']['volatility'],
            'Oracle Score': all_metrics['scores']['oracle'],
            'Composite Score': all_metrics['scores']['composite'],
            'Max OI (USD)': all_metrics['limits']['max_oi_usd'],
            'Max Position (USD)': all_metrics['limits']['max_position_usd'],
            'OI/Volume Ratio': all_metrics['health']['oi_to_volume_ratio'],
            'Health Status': all_metrics['health']['oi_health_status']
        })
    
    return pd.DataFrame(risk_rows)

def run_stress_tests(market_data: Dict[str, Dict], 
                    scenarios_to_run: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Run stress tests for all markets
    
    Args:
        market_data: Dictionary of market data
        scenarios_to_run: List of scenario names to run (None = all)
        
    Returns:
        Dictionary of stress test results by market
    """
    results = {}
    
    default_scenarios = ['liquidation_10pct', 'liquidation_20pct', 
                        'gap_10pct', 'gap_20pct']
    scenarios_to_run = scenarios_to_run or default_scenarios
    
    for name, data in market_data.items():
        all_metrics = metrics.calculate_all_metrics(data)
        max_oi = all_metrics['limits']['max_oi_usd']
        
        market_scenarios = scenarios.run_all_scenarios(data, max_oi)
        
        # Filter to requested scenarios
        filtered_scenarios = {
            k: v for k, v in market_scenarios.items() 
            if any(s in k for s in scenarios_to_run)
        }
        
        results[name] = filtered_scenarios
    
    return results

def format_orderbook_summary(orderbook: Dict) -> Dict[str, any]:
    """
    Format orderbook data for display
    
    Args:
        orderbook: Raw orderbook data
        
    Returns:
        Summary statistics
    """
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    return {
        'bid_levels': len(bids),
        'ask_levels': len(asks),
        'best_bid': bids[0][0] if bids else 0,
        'best_ask': asks[0][0] if asks else 0,
        'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
        'spread_pct': ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if bids and asks and bids[0][0] > 0 else 0,
        'bid_depth_10': sum(b[0] * b[1] for b in bids[:10]) if len(bids) >= 10 else 0,
        'ask_depth_10': sum(a[0] * a[1] for a in asks[:10]) if len(asks) >= 10 else 0
    }

def generate_executive_summary(market_data: Dict[str, Dict]) -> str:
    """
    Generate executive summary text
    
    Args:
        market_data: Dictionary of market data
        
    Returns:
        Markdown formatted summary
    """
    risk_df = calculate_risk_metrics_all(market_data)
    
    # Count risk levels
    high_risk = len(risk_df[risk_df['Composite Score'] >= 4])
    medium_risk = len(risk_df[(risk_df['Composite Score'] >= 3) & 
                              (risk_df['Composite Score'] < 4)])
    low_risk = len(risk_df[risk_df['Composite Score'] < 3])
    
    # Total recommendations
    total_max_oi = risk_df['Max OI (USD)'].sum()
    avg_health_ratio = risk_df['OI/Volume Ratio'].mean()
    
    summary = f"""## Executive Summary

### Market Coverage
- **Total Markets Analyzed**: {len(market_data)}
- **Data Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

### Risk Distribution
- **Low Risk Markets**: {low_risk} (Composite Score < 3)
- **Medium Risk Markets**: {medium_risk} (Composite Score 3-4)
- **High Risk Markets**: {high_risk} (Composite Score ≥ 4)

### Key Recommendations
- **Total Recommended OI Capacity**: ${total_max_oi:,.0f}
- **Average OI/Volume Ratio**: {avg_health_ratio:.2f}x
- **Markets Requiring Attention**: {', '.join(risk_df[risk_df['Health Status'] != 'healthy']['Market'].tolist()) or 'None'}

### Action Items
1. Review position limits for high-risk markets
2. Monitor markets with OI/Volume ratios above 5x
3. Consider increasing liquidity buffers for volatile assets
"""
    
    return summary