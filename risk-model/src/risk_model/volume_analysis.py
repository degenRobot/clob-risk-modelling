"""Volume analysis and correlation module"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def analyze_volume_patterns(klines_df: pd.DataFrame, 
                          window_hours: List[int] = [24, 168, 720]) -> pd.DataFrame:
    """
    Analyze trading volume patterns
    
    Args:
        klines_df: OHLC data with volume
        window_hours: List of rolling window sizes in hours
        
    Returns:
        DataFrame with volume metrics
    """
    results = pd.DataFrame(index=klines_df.index)
    results['volume'] = klines_df['volume'] * klines_df['close']  # USD volume
    results['quote_volume'] = klines_df['quote_volume']
    
    # Calculate rolling averages
    for window in window_hours:
        col_name = f'vol_ma_{window}h'
        results[col_name] = results['volume'].rolling(window).mean()
        
    # Volume profile (distribution by price level)
    try:
        price_bins = pd.qcut(klines_df['close'], q=20, duplicates='drop')
        volume_profile = klines_df.groupby(price_bins, observed=False)['volume'].sum()
    except (ValueError, TypeError):
        # Handle case where all prices are NaN or insufficient unique values
        volume_profile = pd.Series([], dtype=float)
    
    # Volume spikes
    vol_mean = results['volume'].mean()
    vol_std = results['volume'].std()
    results['volume_z_score'] = (results['volume'] - vol_mean) / vol_std
    results['is_volume_spike'] = results['volume_z_score'] > 2
    
    # Time-based patterns
    results['hour'] = results.index.hour
    results['day_of_week'] = results.index.dayofweek
    
    return results, volume_profile

def correlate_volume_depth(volume_df: pd.DataFrame, 
                         depth_metrics: Dict[str, float],
                         lookback_periods: int = 100) -> Dict[str, float]:
    """
    Calculate correlation between volume and depth metrics
    
    Args:
        volume_df: Volume time series
        depth_metrics: Dictionary of depth measurements
        lookback_periods: Number of periods for correlation
        
    Returns:
        Correlation metrics
    """
    correlations = {}
    
    if len(volume_df) < lookback_periods:
        logger.warning("Insufficient data for correlation analysis")
        return correlations
        
    recent_volume = volume_df['volume'].iloc[-lookback_periods:]
    
    # Simple correlation with current depth
    if 'depth_1pct' in depth_metrics:
        # Simulate depth time series (in practice, would fetch historical)
        depth_series = pd.Series(
            [depth_metrics['depth_1pct']] * lookback_periods,
            index=recent_volume.index
        )
        correlations['volume_depth_corr'] = recent_volume.corr(depth_series)
    
    # Volume momentum
    volume_returns = recent_volume.pct_change().dropna()
    correlations['volume_autocorr'] = volume_returns.autocorr(lag=1)
    
    return correlations

def calculate_liquidity_score(volume_24h: float, depth_1pct: float,
                            spread_bps: float) -> Dict[str, float]:
    """
    Calculate composite liquidity score
    
    Args:
        volume_24h: 24-hour trading volume
        depth_1pct: Market depth at 1% impact
        spread_bps: Bid-ask spread in basis points
        
    Returns:
        Liquidity metrics and score
    """
    # Normalize metrics (higher is better)
    volume_score = min(volume_24h / 1e9, 1.0)  # $1B daily volume = max score
    depth_score = min(depth_1pct / 5e7, 1.0)   # $50M depth = max score
    spread_score = max(1.0 - spread_bps / 20, 0)  # 20bps spread = 0 score
    
    # Weighted composite
    weights = {'volume': 0.4, 'depth': 0.4, 'spread': 0.2}
    composite_score = (
        volume_score * weights['volume'] +
        depth_score * weights['depth'] +
        spread_score * weights['spread']
    )
    
    return {
        'volume_score': volume_score,
        'depth_score': depth_score,
        'spread_score': spread_score,
        'composite_liquidity_score': composite_score,
        'liquidity_rating': _get_liquidity_rating(composite_score)
    }

def _get_liquidity_rating(score: float) -> str:
    """Convert liquidity score to rating"""
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.4:
        return 'Fair'
    elif score >= 0.2:
        return 'Poor'
    else:
        return 'Very Poor'

def analyze_volume_volatility_relationship(volume_df: pd.DataFrame,
                                         volatility: float) -> Dict[str, float]:
    """
    Analyze relationship between volume and volatility
    
    Args:
        volume_df: Volume time series
        volatility: Current realized volatility
        
    Returns:
        Analysis metrics
    """
    # Calculate volume volatility
    volume_returns = volume_df['volume'].pct_change().dropna()
    volume_vol = volume_returns.std() * np.sqrt(24 * 365)
    
    # Volume/volatility ratio (activity vs uncertainty)
    avg_volume = volume_df['volume'].mean()
    vol_activity_ratio = avg_volume / (volatility * 1e9) if volatility > 0 else 0
    
    # High volume days vs volatility
    high_vol_days = volume_df[volume_df['volume_z_score'] > 1.5]
    pct_high_vol_days = len(high_vol_days) / len(volume_df) * 100
    
    return {
        'volume_volatility': volume_vol,
        'vol_activity_ratio': vol_activity_ratio,
        'pct_high_volume_days': pct_high_vol_days,
        'volume_consistency': 1 / (1 + volume_vol)  # 0-1 score
    }

def generate_volume_report(market_data: Dict) -> pd.DataFrame:
    """
    Generate comprehensive volume analysis report
    
    Args:
        market_data: Dictionary with all market data
        
    Returns:
        Report DataFrame
    """
    report_data = {
        'metric': [],
        'value': [],
        'benchmark': [],
        'status': []
    }
    
    # Extract data
    volume_24h = market_data.get('volume_24h_usd', 0)
    avg_volume_7d = market_data.get('avg_volume_7d', 0)
    depth_1pct = market_data.get('depth_1pct', 0)
    
    # Volume metrics
    metrics = [
        ('24h Volume (USD)', volume_24h, 1e8, volume_24h > 1e8),
        ('7d Avg Volume (USD)', avg_volume_7d, 1e8, avg_volume_7d > 1e8),
        ('Volume/Depth Ratio', volume_24h/depth_1pct if depth_1pct > 0 else 0, 20, 
         volume_24h/depth_1pct < 20 if depth_1pct > 0 else False),
        ('Volume Volatility', market_data.get('volume_volatility', 0), 0.5,
         market_data.get('volume_volatility', 1) < 0.5)
    ]
    
    for name, value, benchmark, is_good in metrics:
        report_data['metric'].append(name)
        report_data['value'].append(f'{value:,.2f}' if value > 1000 else f'{value:.3f}')
        report_data['benchmark'].append(f'{benchmark:,.0f}' if benchmark > 1000 else f'{benchmark:.1f}')
        report_data['status'].append('✓ Good' if is_good else '⚠ Review')
    
    return pd.DataFrame(report_data)