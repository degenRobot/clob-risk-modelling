"""Stress testing and scenario analysis module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def liquidation_impact_scenario(max_oi_usd: float, liq_curve: pd.DataFrame,
                              liq_fraction: float = 0.2) -> Dict[str, any]:
    """
    Simulate mass liquidation scenario
    
    Args:
        max_oi_usd: Maximum open interest
        liq_curve: Orderbook liquidity curve
        liq_fraction: Fraction of OI to liquidate (0-1)
        
    Returns:
        Dictionary with impact metrics
    """
    liquidation_size = max_oi_usd * liq_fraction
    
    # Find impact for this size on both sides
    bid_impact = 0.0
    ask_impact = 0.0
    
    if not liq_curve.empty:
        # Bid side (sells)
        bid_curve = liq_curve[liq_curve['side'] == 'bid']
        if not bid_curve.empty:
            exceeds = bid_curve[bid_curve['cum_notional'] >= liquidation_size]
            if not exceeds.empty:
                bid_impact = exceeds.iloc[0]['price_impact_pct']
            else:
                bid_impact = bid_curve.iloc[-1]['price_impact_pct']
        
        # Ask side (buys to close shorts)
        ask_curve = liq_curve[liq_curve['side'] == 'ask']
        if not ask_curve.empty:
            exceeds = ask_curve[ask_curve['cum_notional'] >= liquidation_size]
            if not exceeds.empty:
                ask_impact = exceeds.iloc[0]['price_impact_pct']
            else:
                ask_impact = ask_curve.iloc[-1]['price_impact_pct']
    
    avg_impact = (bid_impact + ask_impact) / 2
    
    return {
        "scenario": "mass_liquidation",
        "liquidation_size_usd": liquidation_size,
        "liquidation_pct": liq_fraction * 100,
        "bid_impact_pct": bid_impact,
        "ask_impact_pct": ask_impact,
        "avg_impact_pct": avg_impact,
        "severity": "high" if avg_impact > 5 else "medium" if avg_impact > 2 else "low"
    }

def gap_risk_scenario(current_price: float, gap_pct: float,
                     maintenance_margin: float, oi_usd: float,
                     total_liq_usd: float) -> Dict[str, any]:
    """
    Analyze price gap risk scenario
    
    Args:
        current_price: Current asset price
        gap_pct: Price gap percentage
        maintenance_margin: Maintenance margin requirement (0-1)
        oi_usd: Current open interest
        total_liq_usd: Total available liquidity
        
    Returns:
        Dictionary with gap risk metrics
    """
    gap_price = current_price * (1 - gap_pct / 100)
    price_move_pct = gap_pct
    
    # Positions underwater calculation
    # Assume uniform distribution of entry prices
    positions_underwater_pct = min(100, price_move_pct / (maintenance_margin * 100) * 100)
    
    # Potential liquidations
    potential_liquidations = oi_usd * (positions_underwater_pct / 100)
    
    # Can market absorb liquidations?
    absorption_ratio = potential_liquidations / total_liq_usd if total_liq_usd > 0 else float('inf')
    
    return {
        "scenario": "price_gap",
        "gap_pct": gap_pct,
        "gap_price": gap_price,
        "positions_underwater_pct": positions_underwater_pct,
        "potential_liquidations_usd": potential_liquidations,
        "liquidity_absorption_ratio": absorption_ratio,
        "can_absorb": absorption_ratio < 0.5,
        "severity": "high" if absorption_ratio > 1 else "medium" if absorption_ratio > 0.5 else "low"
    }

def volatility_shock_scenario(current_vol: float, shock_multiplier: float,
                            max_oi_usd: float) -> Dict[str, any]:
    """
    Analyze volatility shock scenario
    
    Args:
        current_vol: Current annualized volatility
        shock_multiplier: Vol increase multiplier (e.g., 2.0 for doubling)
        max_oi_usd: Maximum OI under current conditions
        
    Returns:
        Dictionary with vol shock impact
    """
    shocked_vol = current_vol * shock_multiplier
    
    # Higher vol typically means we need to reduce OI
    # Simple heuristic: OI scales inversely with vol
    adjusted_max_oi = max_oi_usd / shock_multiplier
    oi_reduction_pct = (1 - 1/shock_multiplier) * 100
    
    return {
        "scenario": "volatility_shock",
        "current_vol": current_vol,
        "shocked_vol": shocked_vol,
        "shock_multiplier": shock_multiplier,
        "current_max_oi_usd": max_oi_usd,
        "adjusted_max_oi_usd": adjusted_max_oi,
        "oi_reduction_pct": oi_reduction_pct,
        "severity": "high" if shock_multiplier > 3 else "medium" if shock_multiplier > 2 else "low"
    }

def cascade_scenario(initial_liquidation_pct: float, max_oi_usd: float,
                   liq_curve: pd.DataFrame, rounds: int = 3) -> Dict[str, any]:
    """
    Model cascading liquidations
    
    Args:
        initial_liquidation_pct: Initial liquidation as % of OI
        max_oi_usd: Maximum open interest
        liq_curve: Orderbook liquidity curve
        rounds: Number of cascade rounds to model
        
    Returns:
        Dictionary with cascade analysis
    """
    results = []
    remaining_oi = max_oi_usd
    cumulative_impact = 0
    
    for round_num in range(rounds):
        # Calculate liquidation size for this round
        if round_num == 0:
            liq_size = max_oi_usd * initial_liquidation_pct / 100
        else:
            # Subsequent rounds triggered by price impact
            trigger_threshold = 5  # 5% move triggers more liquidations
            if cumulative_impact > trigger_threshold:
                liq_size = remaining_oi * 0.1  # 10% more liquidated
            else:
                break
        
        # Get impact for this liquidation
        scenario = liquidation_impact_scenario(remaining_oi, liq_curve, 
                                             liq_size / remaining_oi if remaining_oi > 0 else 0)
        
        round_impact = scenario['avg_impact_pct']
        cumulative_impact += round_impact
        remaining_oi -= liq_size
        
        results.append({
            "round": round_num + 1,
            "liquidation_size": liq_size,
            "round_impact_pct": round_impact,
            "cumulative_impact_pct": cumulative_impact,
            "remaining_oi": remaining_oi
        })
    
    return {
        "scenario": "liquidation_cascade",
        "initial_trigger_pct": initial_liquidation_pct,
        "rounds": results,
        "total_liquidated_usd": max_oi_usd - remaining_oi,
        "total_impact_pct": cumulative_impact,
        "severity": "high" if cumulative_impact > 20 else "medium" if cumulative_impact > 10 else "low"
    }

def run_all_scenarios(market_data: Dict, max_oi_usd: float) -> Dict[str, any]:
    """
    Run all stress test scenarios
    
    Args:
        market_data: Market data including liquidity curves
        max_oi_usd: Maximum OI to test
        
    Returns:
        Dictionary with all scenario results
    """
    liq_curve = market_data.get('liquidity_curve', pd.DataFrame())
    current_price = market_data.get('current_price', 100)
    current_vol = market_data.get('realized_vol', 0.5)
    total_liq = market_data.get('total_liquidity_usd', 0)
    
    scenarios = {}
    
    # Mass liquidation scenarios
    for liq_pct in [0.1, 0.2, 0.4]:
        scenario = liquidation_impact_scenario(max_oi_usd, liq_curve, liq_pct)
        scenarios[f"liquidation_{int(liq_pct*100)}pct"] = scenario
    
    # Gap risk scenarios  
    for gap in [5, 10, 20]:
        scenario = gap_risk_scenario(current_price, gap, 0.05, max_oi_usd, total_liq)
        scenarios[f"gap_{gap}pct"] = scenario
        
    # Volatility shocks
    for shock in [2, 3, 5]:
        scenario = volatility_shock_scenario(current_vol, shock, max_oi_usd)
        scenarios[f"vol_shock_{shock}x"] = scenario
        
    # Cascade scenario
    cascade = cascade_scenario(10, max_oi_usd, liq_curve)
    scenarios["cascade_10pct"] = cascade
    
    return scenarios

def summarize_scenarios(scenarios: Dict[str, any]) -> pd.DataFrame:
    """
    Summarize scenario results in a DataFrame
    
    Args:
        scenarios: Dictionary of scenario results
        
    Returns:
        Summary DataFrame
    """
    rows = []
    
    for name, scenario in scenarios.items():
        row = {
            "scenario": name,
            "type": scenario.get("scenario", "unknown"),
            "severity": scenario.get("severity", "unknown")
        }
        
        # Add type-specific metrics
        if "impact_pct" in scenario:
            row["impact_pct"] = scenario.get("avg_impact_pct", scenario.get("impact_pct", 0))
        elif "total_impact_pct" in scenario:
            row["impact_pct"] = scenario["total_impact_pct"]
        else:
            row["impact_pct"] = 0
            
        rows.append(row)
    
    return pd.DataFrame(rows)