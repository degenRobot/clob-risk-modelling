"""Pricing and liquidity analysis module"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def compute_orderbook_liquidity_curve(orderbook: Dict, 
                                    max_impact_pct: float = 5.0) -> pd.DataFrame:
    """
    Build cumulative liquidity curves from orderbook data
    
    Args:
        orderbook: Dictionary with 'bids' and 'asks' arrays
        max_impact_pct: Maximum price impact to calculate up to
        
    Returns:
        DataFrame with columns: side, price, qty, cum_qty, cum_notional, price_impact_pct
    """
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    # Calculate mid price
    if len(bids) > 0 and len(asks) > 0:
        mid_price = (bids[0][0] + asks[0][0]) / 2
    else:
        logger.warning("Empty orderbook")
        return pd.DataFrame()
    
    # Process bid side (sells into bids)
    bid_data = []
    cum_qty = 0
    cum_notional = 0
    
    for price, qty in bids:
        cum_qty += qty
        cum_notional += price * qty
        impact_pct = ((mid_price - price) / mid_price) * 100
        
        if impact_pct > max_impact_pct:
            break
            
        bid_data.append({
            'side': 'bid',
            'price': price,
            'qty': qty,
            'cum_qty': cum_qty,
            'cum_notional': cum_notional,
            'price_impact_pct': impact_pct
        })
    
    # Process ask side (buys from asks)
    ask_data = []
    cum_qty = 0
    cum_notional = 0
    
    for price, qty in asks:
        cum_qty += qty
        cum_notional += price * qty
        impact_pct = ((price - mid_price) / mid_price) * 100
        
        if impact_pct > max_impact_pct:
            break
            
        ask_data.append({
            'side': 'ask',
            'price': price,
            'qty': qty,
            'cum_qty': cum_qty,
            'cum_notional': cum_notional,
            'price_impact_pct': impact_pct
        })
    
    # Combine into DataFrame
    df = pd.DataFrame(bid_data + ask_data)
    
    if not df.empty:
        df['mid_price'] = mid_price
        
    return df

def depth_at_impact(curve: pd.DataFrame, max_impact_pct: float, side: str = 'both') -> float:
    """
    Calculate maximum notional available at given price impact
    
    Args:
        curve: Liquidity curve from compute_orderbook_liquidity_curve
        max_impact_pct: Maximum acceptable price impact
        side: 'bid', 'ask', or 'both'
        
    Returns:
        Maximum notional in quote currency (e.g., USD)
    """
    if curve.empty:
        return 0.0
    
    if side == 'both':
        bid_depth = depth_at_impact(curve, max_impact_pct, 'bid')
        ask_depth = depth_at_impact(curve, max_impact_pct, 'ask')
        return (bid_depth + ask_depth) / 2
    
    side_curve = curve[curve['side'] == side]
    valid_levels = side_curve[side_curve['price_impact_pct'] <= max_impact_pct]
    
    if valid_levels.empty:
        return 0.0
        
    return valid_levels['cum_notional'].max()

def realized_vol(klines_df: pd.DataFrame, window: int = 30*24, 
                freq_per_day: int = 24) -> float:
    """
    Calculate realized volatility from OHLC data
    
    Args:
        klines_df: DataFrame with OHLC data
        window: Number of periods for calculation
        freq_per_day: Periods per day (24 for hourly, 1 for daily)
        
    Returns:
        Annualized realized volatility (0-1 scale)
    """
    if len(klines_df) < window:
        # Silently adjust window to available data
        window = len(klines_df)
    
    # Calculate log returns
    close_prices = klines_df['close'].iloc[-window:]
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    if len(log_returns) < 2:
        return 0.0
    
    # Calculate standard deviation
    std_dev = log_returns.std()
    
    # Annualize
    periods_per_year = freq_per_day * 365
    annualized_vol = std_dev * np.sqrt(periods_per_year)
    
    return annualized_vol

def garch_vol(klines_df: pd.DataFrame, forecast_horizon: int = 24) -> Dict[str, float]:
    """
    Simple GARCH-like volatility forecast
    
    Args:
        klines_df: DataFrame with OHLC data  
        forecast_horizon: Hours to forecast ahead
        
    Returns:
        Dictionary with current vol and forecast
    """
    # Calculate returns
    returns = klines_df['close'].pct_change().dropna()
    
    # Simple EWMA approach (not full GARCH)
    ewma_vol = returns.ewm(span=24).std().iloc[-1]
    
    # Annualize
    annualized_current = ewma_vol * np.sqrt(365 * 24)
    
    # Simple forecast (persistence)
    forecast_vol = annualized_current
    
    return {
        'current_vol': annualized_current,
        'forecast_vol': forecast_vol,
        'ewma_hourly_vol': ewma_vol
    }

def combine_liquidity_sources(cex_depth_usd: float, amm_depth_usd: float,
                            amm_haircut: float = 0.7) -> Dict[str, float]:
    """
    Combine CEX and AMM liquidity with appropriate haircuts
    
    Args:
        cex_depth_usd: CEX liquidity at target impact
        amm_depth_usd: AMM liquidity at target impact  
        amm_haircut: Haircut factor for AMM liquidity (0-1)
        
    Returns:
        Dictionary with combined liquidity metrics
    """
    adjusted_amm = amm_depth_usd * amm_haircut
    total_liquidity = cex_depth_usd + adjusted_amm
    
    return {
        'cex_liquidity_usd': cex_depth_usd,
        'amm_liquidity_usd': amm_depth_usd,
        'amm_adjusted_usd': adjusted_amm,
        'total_liquidity_usd': total_liquidity,
        'cex_share': cex_depth_usd / total_liquidity if total_liquidity > 0 else 0,
        'amm_share': adjusted_amm / total_liquidity if total_liquidity > 0 else 0
    }

def calculate_spread_metrics(book_ticker: Dict) -> Dict[str, float]:
    """
    Calculate spread metrics from best bid/ask
    
    Args:
        book_ticker: Dictionary with bidPrice, askPrice, etc.
        
    Returns:
        Dictionary with spread metrics
    """
    try:
        bid_price = float(book_ticker.get('bidPrice', 0))
        ask_price = float(book_ticker.get('askPrice', 0))
        bid_qty = float(book_ticker.get('bidQty', 0))
        ask_qty = float(book_ticker.get('askQty', 0))
        
        if bid_price > 0 and ask_price > 0:
            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            spread_bps = (spread / mid_price) * 10000
            
            return {
                'mid_price': mid_price,
                'spread': spread,
                'spread_bps': spread_bps,
                'bid_size': bid_qty * bid_price,
                'ask_size': ask_qty * ask_price
            }
    except Exception as e:
        logger.error(f"Failed to calculate spread metrics: {e}")
        
    return {
        'mid_price': 0,
        'spread': 0,
        'spread_bps': 0,
        'bid_size': 0,
        'ask_size': 0
    }

def estimate_execution_cost(size_usd: float, liquidity_curve: pd.DataFrame,
                          spread_bps: float) -> Dict[str, float]:
    """
    Estimate total execution cost including spread and impact
    
    Args:
        size_usd: Trade size in USD
        liquidity_curve: Orderbook liquidity curve
        spread_bps: Current spread in basis points
        
    Returns:
        Dictionary with cost breakdown
    """
    # Find price impact for this size
    impact_pct = 0.0
    
    for side in ['bid', 'ask']:
        side_curve = liquidity_curve[liquidity_curve['side'] == side]
        if not side_curve.empty:
            # Find where cumulative notional exceeds our size
            exceeds = side_curve[side_curve['cum_notional'] >= size_usd]
            if not exceeds.empty:
                impact_pct = max(impact_pct, exceeds.iloc[0]['price_impact_pct'])
    
    # Convert spread cost
    spread_cost_pct = spread_bps / 10000
    
    # Total cost
    total_cost_pct = spread_cost_pct + impact_pct
    total_cost_usd = size_usd * (total_cost_pct / 100)
    
    return {
        'spread_cost_pct': spread_cost_pct,
        'impact_cost_pct': impact_pct,
        'total_cost_pct': total_cost_pct,
        'total_cost_usd': total_cost_usd,
        'effective_price_mult': 1 + (total_cost_pct / 100)
    }