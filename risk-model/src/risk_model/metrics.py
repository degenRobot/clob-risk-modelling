"""Risk metrics and scoring module"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Risk score mappings
RISK_TO_MAX_OI = {
    1: 0.5,    # Very safe - 50% of available liquidity
    2: 0.35,   # Safe - 35%  
    3: 0.25,   # Medium - 25%
    4: 0.15,   # Risky - 15%
    5: 0.07,   # Very risky - 7%
}

# Impact to position mappings
IMPACT_TO_MAX_POSITION = {
    0.01: 0.5,   # 1% impact - 50% of max OI
    0.02: 0.35,  # 2% impact - 35% of max OI
    0.05: 0.20,  # 5% impact - 20% of max OI
    0.10: 0.10,  # 10% impact - 10% of max OI
}

def liquidity_score(total_liq_usd: float) -> int:
    """
    Score liquidity depth from 1 (best) to 5 (worst)
    
    Args:
        total_liq_usd: Total available liquidity at 1% impact
        
    Returns:
        Score from 1-5
    """
    if total_liq_usd >= 50_000_000:
        return 1  # Very deep
    elif total_liq_usd >= 20_000_000:
        return 2  # Deep
    elif total_liq_usd >= 5_000_000:
        return 3  # Medium
    elif total_liq_usd >= 1_000_000:
        return 4  # Shallow
    else:
        return 5  # Very shallow

def volatility_score(realized_vol: float) -> int:
    """
    Score volatility from 1 (lowest) to 5 (highest)
    
    Args:
        realized_vol: Annualized realized volatility (0-1 scale)
        
    Returns:
        Score from 1-5
    """
    vol_pct = realized_vol * 100
    
    if vol_pct <= 30:
        return 1  # Very low vol
    elif vol_pct <= 50:
        return 2  # Low vol
    elif vol_pct <= 80:
        return 3  # Medium vol
    elif vol_pct <= 120:
        return 4  # High vol
    else:
        return 5  # Very high vol

def oracle_score(asset: str, oracle_meta: Optional[Dict] = None) -> int:
    """
    Score oracle quality/reliability from 1 (best) to 5 (worst)
    
    Args:
        asset: Asset symbol
        oracle_meta: Optional oracle metadata
        
    Returns:
        Score from 1-5
    """
    # Default scoring based on known oracles
    oracle_type = oracle_meta.get('type', '') if oracle_meta else ''
    
    # Major assets with Chainlink typically score well
    major_assets = ['ETH', 'BTC', 'LINK', 'AAVE', 'UNI']
    if any(asset.upper().startswith(m) for m in major_assets):
        if oracle_type.lower() == 'chainlink':
            return 1
        return 2
    
    # Other chainlink oracles
    if oracle_type.lower() == 'chainlink':
        return 2
        
    # Pyth oracles
    if oracle_type.lower() == 'pyth':
        return 3
        
    # Unknown or custom oracles
    return 4

def composite_risk_score(liq_score: int, vol_score: int, oracle_score: int,
                       weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> int:
    """
    Calculate weighted composite risk score
    
    Args:
        liq_score: Liquidity score (1-5)
        vol_score: Volatility score (1-5)
        oracle_score: Oracle quality score (1-5)
        weights: Weights for (liquidity, volatility, oracle)
        
    Returns:
        Composite score rounded to 1-5
    """
    w_liq, w_vol, w_oracle = weights
    
    # Normalize weights
    total_weight = w_liq + w_vol + w_oracle
    w_liq /= total_weight
    w_vol /= total_weight  
    w_oracle /= total_weight
    
    # Calculate weighted average
    weighted_score = (liq_score * w_liq + 
                     vol_score * w_vol + 
                     oracle_score * w_oracle)
    
    # Round to nearest integer 1-5
    return int(np.clip(np.round(weighted_score), 1, 5))

def recommend_max_oi(total_liq_usd: float, risk_score: int) -> float:
    """
    Recommend maximum open interest based on liquidity and risk
    
    Args:
        total_liq_usd: Total available liquidity
        risk_score: Composite risk score (1-5)
        
    Returns:
        Recommended max OI in USD
    """
    ratio = RISK_TO_MAX_OI.get(risk_score, 0.07)
    return total_liq_usd * ratio

def recommend_max_position(max_oi_usd: float, single_pos_pct: float = 0.15) -> float:
    """
    Recommend maximum single position size
    
    Args:
        max_oi_usd: Maximum total OI
        single_pos_pct: Max single position as % of total OI
        
    Returns:
        Recommended max position size in USD
    """
    return max_oi_usd * single_pos_pct

def oi_to_volume_ratio(max_oi_usd: float, volume_24h_usd: float) -> float:
    """
    Calculate OI to daily volume ratio
    
    Args:
        max_oi_usd: Maximum or current OI
        volume_24h_usd: 24-hour trading volume
        
    Returns:
        Ratio of OI to volume
    """
    if volume_24h_usd <= 0:
        return float('inf')
    return max_oi_usd / volume_24h_usd

def evaluate_oi_health(oi_volume_ratio: float, warning_threshold: float = 5.0) -> Dict[str, any]:
    """
    Evaluate health of OI relative to volume
    
    Args:
        oi_volume_ratio: OI to volume ratio
        warning_threshold: Threshold for warning
        
    Returns:
        Dictionary with health status
    """
    if oi_volume_ratio < 1.0:
        status = "healthy"
        message = "OI well supported by volume"
    elif oi_volume_ratio < warning_threshold:
        status = "caution"
        message = f"OI is {oi_volume_ratio:.1f}x daily volume"
    else:
        status = "warning"
        message = f"OI is {oi_volume_ratio:.1f}x daily volume - consider reducing"
        
    return {
        "status": status,
        "ratio": oi_volume_ratio,
        "message": message,
        "is_healthy": oi_volume_ratio < warning_threshold
    }

def calculate_all_metrics(market_data: Dict) -> Dict[str, any]:
    """
    Calculate all risk metrics for a market
    
    Args:
        market_data: Dictionary with liquidity, volatility, volume data
        
    Returns:
        Comprehensive metrics dictionary
    """
    # Extract data
    total_liq = market_data.get('total_liquidity_usd', 0)
    vol = market_data.get('realized_vol', 0.5)
    volume_24h = market_data.get('volume_24h_usd', 0)
    asset = market_data.get('asset', 'UNKNOWN')
    oracle_meta = market_data.get('oracle_meta', {})
    
    # Calculate scores
    liq_s = liquidity_score(total_liq)
    vol_s = volatility_score(vol)
    orc_s = oracle_score(asset, oracle_meta)
    
    # Composite score
    risk_s = composite_risk_score(liq_s, vol_s, orc_s)
    
    # Recommendations
    max_oi = recommend_max_oi(total_liq, risk_s)
    max_pos = recommend_max_position(max_oi)
    
    # OI health
    oi_vol_ratio = oi_to_volume_ratio(max_oi, volume_24h)
    oi_health = evaluate_oi_health(oi_vol_ratio)
    
    return {
        "scores": {
            "liquidity": liq_s,
            "volatility": vol_s,
            "oracle": orc_s,
            "composite": risk_s
        },
        "limits": {
            "max_oi_usd": max_oi,
            "max_position_usd": max_pos,
            "max_oi_pct_of_liquidity": RISK_TO_MAX_OI[risk_s] * 100
        },
        "health": {
            "oi_to_volume_ratio": oi_vol_ratio,
            "oi_health_status": oi_health["status"],
            "oi_health_message": oi_health["message"]
        },
        "inputs": {
            "total_liquidity_usd": total_liq,
            "realized_vol": vol,
            "volume_24h_usd": volume_24h
        }
    }

def calculate_position_limits(liquidity_usd: float, volatility: float, risk_score: int) -> Dict[str, float]:
    """
    Calculate position limits based on market conditions
    
    Args:
        liquidity_usd: Available liquidity in USD
        volatility: Market volatility (annualized)
        risk_score: Composite risk score (1-5)
        
    Returns:
        Dictionary with various position limits
    """
    # Base limit from liquidity
    base_limit = liquidity_usd * 0.1  # 10% of liquidity as base
    
    # Volatility adjustment
    if volatility < 0.5:  # < 50% annual vol
        vol_multiplier = 1.0
    elif volatility < 1.0:  # 50-100% vol
        vol_multiplier = 0.75
    else:  # > 100% vol
        vol_multiplier = 0.5
    
    # Risk score adjustment
    risk_multiplier = RISK_TO_MAX_OI.get(risk_score, 0.07)
    
    # Calculate final limits
    position_limit = base_limit * vol_multiplier * risk_multiplier
    
    return {
        "position_limit": position_limit,
        "base_limit": base_limit,
        "vol_adjustment": vol_multiplier,
        "risk_adjustment": risk_multiplier,
        "tier_limits": {
            "retail": position_limit * 0.1,
            "professional": position_limit * 0.25,
            "market_maker": position_limit * 0.5,
            "institutional": position_limit
        }
    }

def calculate_max_oi(liquidity_usd: float, risk_score: int) -> float:
    """
    Calculate maximum open interest based on liquidity and risk
    
    Args:
        liquidity_usd: Total available liquidity in USD
        risk_score: Composite risk score (1-5)
        
    Returns:
        Maximum recommended open interest in USD
    """
    return recommend_max_oi(liquidity_usd, risk_score)