"""Stress testing and scenario analysis module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def simulate_liquidation_cascade(orderbook: Dict, liquidation_pct: float = 0.2) -> Dict[str, any]:
    """Simulate liquidation cascade impact"""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {"error": "Invalid orderbook data"}
    
    mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
    total_bid_depth = sum(p * s for p, s in orderbook['bids'][:50])
    
    # Simulate selling pressure from liquidations
    liquidation_size = total_bid_depth * liquidation_pct
    price_impact = 0
    remaining_size = liquidation_size
    
    for price, size in orderbook['bids']:
        if remaining_size <= 0:
            break
        filled = min(remaining_size, size * price)
        remaining_size -= filled
        price_impact = (mid_price - price) / mid_price * 100
    
    # Recovery calculation
    recovery_bps = price_impact * 10  # Simplified recovery metric
    
    return {
        "liquidation_pct": liquidation_pct * 100,
        "price_impact_pct": price_impact,
        "max_price_drop": price_impact,
        "recovery_bps": recovery_bps,
        "market_depth_1pct": total_bid_depth / 100
    }

def simulate_liquidity_gap(orderbook: Dict, gap_size: float, direction: str = 'down') -> Dict[str, any]:
    """Simulate sudden liquidity gap"""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {"error": "Invalid orderbook data"}
    
    orders = orderbook['bids'] if direction == 'down' else orderbook['asks']
    mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
    gap_price = mid_price * (1 - gap_size) if direction == 'down' else mid_price * (1 + gap_size)
    
    unfilled_orders = 0
    for price, size in orders:
        if (direction == 'down' and price > gap_price) or (direction == 'up' and price < gap_price):
            unfilled_orders += 1
    
    return {
        "gap_size": gap_size,
        "direction": direction,
        "unfilled_orders": unfilled_orders,
        "gap_price": gap_price,
        "mid_price": mid_price
    }

def analyze_funding_stress(market_data: Dict, funding_rate: float, periods: int) -> Dict[str, any]:
    """Analyze impact of sustained funding rates"""
    position_value = 1000000  # $1M position
    total_cost = position_value * funding_rate * periods
    total_cost_pct = (total_cost / position_value) * 100
    
    return {
        "funding_rate": funding_rate,
        "periods": periods,
        "total_cost": total_cost,
        "total_cost_pct": total_cost_pct
    }

def generate_scenario_report(scenarios: Dict) -> pd.DataFrame:
    """Generate summary report of scenarios"""
    results = []
    for name, scenario in scenarios.items():
        if isinstance(scenario, dict) and 'error' not in scenario:
            results.append({
                'scenario': name,
                'impact': scenario.get('price_impact_pct', 0),
                'severity': scenario.get('severity', 'unknown')
            })
    return pd.DataFrame(results)

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


def calculate_manipulation_cost(orderbook: Dict, target_impact_pct: float, 
                               side: str = 'up') -> Dict[str, any]:
    """
    Calculate cost to manipulate price by target percentage
    
    Args:
        orderbook: Orderbook data with bids/asks
        target_impact_pct: Desired price impact percentage
        side: Direction of manipulation ('up' or 'down')
        
    Returns:
        Dictionary with manipulation cost analysis
    """
    if side == 'up':
        # Buy through asks to push price up
        orders = orderbook.get('asks', [])
    else:
        # Sell through bids to push price down
        orders = orderbook.get('bids', [])
    
    if len(orders) == 0:
        return {"error": "No orderbook data"}
    
    mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
    target_price = mid_price * (1 + target_impact_pct/100) if side == 'up' else mid_price * (1 - target_impact_pct/100)
    
    total_cost = 0
    total_size = 0
    orders_consumed = 0
    
    for price, size in orders:
        if (side == 'up' and price > target_price) or (side == 'down' and price < target_price):
            break
            
        order_cost = price * size
        total_cost += order_cost
        total_size += size
        orders_consumed += 1
    
    avg_exec_price = total_cost / total_size if total_size > 0 else mid_price
    actual_impact = abs((avg_exec_price - mid_price) / mid_price) * 100
    
    return {
        "target_impact_pct": target_impact_pct,
        "actual_impact_pct": actual_impact,
        "manipulation_cost_usd": total_cost,
        "size_required": total_size,
        "orders_consumed": orders_consumed,
        "avg_execution_price": avg_exec_price,
        "mid_price": mid_price,
        "side": side
    }


def analyze_manipulation_pnl(manipulation_cost: float, price_impact_pct: float,
                           leverage: float, position_size_usd: float,
                           funding_rate: float = 0.01) -> Dict[str, any]:
    """
    Analyze potential PnL from price manipulation with leveraged position
    
    Args:
        manipulation_cost: Cost to manipulate price
        price_impact_pct: Price impact achieved
        leverage: Leverage used on position  
        position_size_usd: Size of leveraged position
        funding_rate: Funding rate per period
        
    Returns:
        Dictionary with PnL analysis
    """
    # Position PnL from price movement
    position_pnl = position_size_usd * (price_impact_pct / 100)
    
    # Cost components
    margin_required = position_size_usd / leverage
    funding_cost = position_size_usd * funding_rate
    
    # Net PnL calculation
    gross_profit = position_pnl - manipulation_cost
    net_profit = gross_profit - funding_cost
    
    # ROI calculations
    total_capital_required = manipulation_cost + margin_required
    roi_pct = (net_profit / total_capital_required) * 100 if total_capital_required > 0 else 0
    
    # Break-even analysis
    breakeven_periods = manipulation_cost / (position_size_usd * funding_rate) if funding_cost > 0 else float('inf')
    
    return {
        "manipulation_cost": manipulation_cost,
        "position_size": position_size_usd,
        "leverage": leverage,
        "margin_required": margin_required,
        "price_impact_pct": price_impact_pct,
        "position_pnl": position_pnl,
        "funding_cost_per_period": funding_cost,
        "gross_profit": gross_profit,
        "net_profit": net_profit,
        "total_capital_required": total_capital_required,
        "roi_pct": roi_pct,
        "breakeven_periods": breakeven_periods,
        "profitable": net_profit > 0
    }


def analyze_funding_manipulation(orderbook: Dict, position_imbalance_usd: float,
                               current_funding: float, max_funding: float = 0.01) -> Dict[str, any]:
    """
    Analyze how position imbalance affects funding rates
    
    Args:
        orderbook: Current orderbook
        position_imbalance_usd: Size of position imbalance to create
        current_funding: Current funding rate
        max_funding: Maximum possible funding rate
        
    Returns:
        Dictionary with funding impact analysis
    """
    mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
    
    # Simple model: funding scales with imbalance
    # In reality this is more complex and exchange-specific
    total_liquidity = sum(p*s for p,s in orderbook['bids'][:20]) + sum(p*s for p,s in orderbook['asks'][:20])
    imbalance_ratio = position_imbalance_usd / total_liquidity if total_liquidity > 0 else 0
    
    # Funding impact (simplified model)
    funding_multiplier = 1 + (imbalance_ratio * 10)  # 10x sensitivity factor
    new_funding = min(current_funding * funding_multiplier, max_funding)
    funding_increase = new_funding - current_funding
    
    # Cost to maintain imbalance
    daily_funding_cost = position_imbalance_usd * new_funding * 3  # 3 funding periods per day
    
    return {
        "position_imbalance_usd": position_imbalance_usd,
        "total_liquidity": total_liquidity,
        "imbalance_ratio": imbalance_ratio * 100,
        "current_funding_rate": current_funding,
        "new_funding_rate": new_funding,
        "funding_increase": funding_increase,
        "daily_funding_cost": daily_funding_cost,
        "funding_multiplier": funding_multiplier
    }


def analyze_oracle_defense(spot_venues: List[Dict], manipulation_impact_pct: float) -> Dict[str, any]:
    """
    Analyze how oracle design defends against manipulation
    
    Args:
        spot_venues: List of spot venues with liquidity info
        manipulation_impact_pct: Attempted manipulation percentage
        
    Returns:
        Dictionary with oracle defense analysis
    """
    total_liquidity = sum(v.get('liquidity_usd', 0) for v in spot_venues)
    venue_count = len(spot_venues)
    
    # Cost to manipulate all venues
    total_manipulation_cost = 0
    manipulatable_venues = 0
    
    for venue in spot_venues:
        venue_liq = venue.get('liquidity_usd', 0)
        venue_cost = venue_liq * (manipulation_impact_pct / 100)
        
        if venue_cost < venue_liq * 0.5:  # Can manipulate if cost < 50% of liquidity
            total_manipulation_cost += venue_cost
            manipulatable_venues += 1
    
    # Oracle resistance calculation
    manipulation_difficulty = "high" if manipulatable_venues < venue_count/2 else "medium" if manipulatable_venues < venue_count else "low"
    
    # Time-weighted average price (TWAP) defense
    twap_window = 300  # 5 minute TWAP
    blocks_needed = twap_window / 12  # ~12 second blocks
    sustained_cost = total_manipulation_cost * (blocks_needed / 10)  # Cost increases with time
    
    return {
        "spot_venues": venue_count,
        "total_spot_liquidity": total_liquidity,
        "manipulatable_venues": manipulatable_venues,
        "manipulation_cost_single": total_manipulation_cost / venue_count if venue_count > 0 else 0,
        "manipulation_cost_all": total_manipulation_cost,
        "twap_window_seconds": twap_window,
        "sustained_manipulation_cost": sustained_cost,
        "manipulation_difficulty": manipulation_difficulty,
        "oracle_resistant": manipulatable_venues < venue_count / 2
    }

# Export all functions
__all__ = [
    'simulate_liquidation_cascade',
    'simulate_liquidity_gap', 
    'analyze_funding_stress',
    'generate_scenario_report',
    'liquidation_impact_scenario',
    'gap_risk_scenario',
    'volatility_shock_scenario',
    'cascade_scenario',
    'run_all_scenarios',
    'summarize_scenarios',
    'calculate_manipulation_cost',
    'analyze_manipulation_pnl',
    'analyze_funding_manipulation',
    'analyze_oracle_defense'
]