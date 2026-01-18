"""
RISEx Contract Parameters and Formulas

This module contains risk parameters and formulas that match the
RISEx smart contracts (risex-contracts/).

Reference: risex-contracts/src/libraries/Constant.sol
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

# =============================================================================
# CONTRACT CONSTANTS (from Constant.sol)
# =============================================================================

# Order Limits
MAX_OPEN_ORDERS = 50  # Per market per account
MAX_CLEAN_UP_ORDERS = 100

# Liquidation Parameters
HEALTH_FACTOR_LIQUIDATION_THRESHOLD = 1.0  # 100%
CLOSEOUT_MARGIN_FACTOR = 2/3  # 66.67% - below this, RLP takes over
LIQUIDATION_FEE_BPS = 100  # 1% of position notional
LIQUIDATION_PRICE_BAND_BPS = 500  # 5% price band around mid
MAX_LIQUIDATION_DEPTH_BPS = 1000  # 10% of available liquidity

# Funding Rate Parameters
SETTLE_FUNDING_RATE_INTERVAL = 3600  # 1 hour in seconds
SNAPSHOT_PREMIUM_INDEX_INTERVAL = 5  # 5 seconds
IMPACT_NOTIONAL_USD = 500  # USD for premium index calculation

# Funding Rate Clamps
INTEREST_RATE = 0.0001  # 0.01% hourly base rate
INTEREST_RATE_MIN_CLAMP = -0.0005  # -0.05%
INTEREST_RATE_MAX_CLAMP = 0.0005  # 0.05%
FUNDING_FLOOR = -0.04  # -4% per hour max
FUNDING_CAP = 0.04  # 4% per hour max


@dataclass
class MarketConfig:
    """Market configuration matching RISEx MarketConfig struct"""
    name: str
    max_leverage: int
    maintenance_margin_factor: float  # e.g., 0.05 = 5%
    min_order_size: float
    step_size: float
    step_price: float
    open_interest_limit: float


# Default market configs
DEFAULT_MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "BTC-PERP": MarketConfig(
        name="BTC-PERP",
        max_leverage=20,
        maintenance_margin_factor=0.05,  # 5% = 20x max leverage
        min_order_size=0.001,
        step_size=0.001,
        step_price=0.01,
        open_interest_limit=50_000_000,  # $50M
    ),
    "ETH-PERP": MarketConfig(
        name="ETH-PERP",
        max_leverage=20,
        maintenance_margin_factor=0.05,
        min_order_size=0.01,
        step_size=0.01,
        step_price=0.01,
        open_interest_limit=50_000_000,
    ),
    "SOL-PERP": MarketConfig(
        name="SOL-PERP",
        max_leverage=10,
        maintenance_margin_factor=0.10,  # 10% = 10x max leverage
        min_order_size=0.1,
        step_size=0.1,
        step_price=0.001,
        open_interest_limit=20_000_000,
    ),
}


# =============================================================================
# MARGIN CALCULATIONS (from OrdersManager.sol)
# =============================================================================

def initial_margin(position_size: float, mark_price: float, leverage: float) -> float:
    """
    Calculate initial margin required.

    Formula: Initial Margin = (Position Size * Mark Price) / Leverage

    Args:
        position_size: Absolute position size in base asset
        mark_price: Current mark price
        leverage: User's chosen leverage (1-20x typically)

    Returns:
        Required initial margin in quote currency (USD)
    """
    notional = abs(position_size) * mark_price
    return notional / leverage


def maintenance_margin(position_size: float, mark_price: float,
                       mm_factor: float) -> float:
    """
    Calculate maintenance margin threshold.

    Formula: MM = (Position Size * Mark Price) * MM_Factor

    Args:
        position_size: Absolute position size in base asset
        mark_price: Current mark price
        mm_factor: Maintenance margin factor (e.g., 0.05 for 5%)

    Returns:
        Maintenance margin threshold in USD
    """
    notional = abs(position_size) * mark_price
    return notional * mm_factor


def is_liquidatable(margin_balance: float, position_size: float,
                   mark_price: float, mm_factor: float) -> bool:
    """
    Check if position should be liquidated.

    Liquidation occurs when: margin_balance < maintenance_margin

    Args:
        margin_balance: Current margin balance (collateral + PnL)
        position_size: Position size
        mark_price: Current mark price
        mm_factor: Maintenance margin factor

    Returns:
        True if position should be liquidated
    """
    mm = maintenance_margin(position_size, mark_price, mm_factor)
    return margin_balance < mm


def is_closeout(margin_balance: float, position_size: float,
                mark_price: float, mm_factor: float) -> bool:
    """
    Check if position should be taken over by RLP (closeout).

    Closeout occurs when: margin_balance < (2/3) * maintenance_margin

    Args:
        margin_balance: Current margin balance
        position_size: Position size
        mark_price: Current mark price
        mm_factor: Maintenance margin factor

    Returns:
        True if RLP should take over position
    """
    mm = maintenance_margin(position_size, mark_price, mm_factor)
    return margin_balance < mm * CLOSEOUT_MARGIN_FACTOR


def liquidation_price(entry_price: float, position_size: float,
                      margin: float, mm_factor: float,
                      is_long: bool) -> float:
    """
    Calculate price at which position gets liquidated.

    For longs: liquidation when price drops enough that margin = MM
    For shorts: liquidation when price rises enough

    Args:
        entry_price: Entry price of position
        position_size: Absolute position size
        margin: Available margin
        mm_factor: Maintenance margin factor
        is_long: True if long position

    Returns:
        Liquidation price
    """
    # Simplified formula assuming isolated margin
    # margin - (entry - liq_price) * size = mm_factor * liq_price * size
    # Solving for liq_price:

    if is_long:
        # Long: price drops
        liq_price = (margin - entry_price * position_size) / (position_size * (mm_factor - 1))
    else:
        # Short: price rises
        liq_price = (margin + entry_price * position_size) / (position_size * (1 + mm_factor))

    return max(0, liq_price)


# =============================================================================
# FUNDING RATE CALCULATIONS (from FundingRate.sol)
# =============================================================================

def premium_index(impact_bid_price: float, impact_ask_price: float,
                 index_price: float) -> float:
    """
    Calculate premium index for funding rate.

    Formula: Premium = [max(0, ImpactBid - Index) - max(0, Index - ImpactAsk)] / Index

    Args:
        impact_bid_price: Execution price for 500 USDC sell
        impact_ask_price: Execution price for 500 USDC buy
        index_price: Oracle index price

    Returns:
        Premium index (can be positive or negative)
    """
    bid_premium = max(0, impact_bid_price - index_price)
    ask_premium = max(0, index_price - impact_ask_price)
    return (bid_premium - ask_premium) / index_price if index_price > 0 else 0


def funding_rate(avg_premium_index: float) -> float:
    """
    Calculate funding rate from average premium index.

    Formula:
        InterestComponent = clamp(0.01% - AvgPremium, [-0.05%, 0.05%])
        FundingRate = (AvgPremium + InterestComponent) / 8
        FundingRate = clamp(FundingRate, [-4%, 4%])

    Args:
        avg_premium_index: Average premium index over funding period

    Returns:
        Funding rate (as decimal, e.g., 0.01 = 1%)
    """
    # Interest rate component
    interest_component = INTEREST_RATE - avg_premium_index
    interest_component = np.clip(interest_component,
                                  INTEREST_RATE_MIN_CLAMP,
                                  INTEREST_RATE_MAX_CLAMP)

    # Raw funding rate (divide by 8 for hourly settlement)
    raw_funding = (avg_premium_index + interest_component) / 8

    # Apply floor/cap
    return np.clip(raw_funding, FUNDING_FLOOR, FUNDING_CAP)


def funding_payment(position_size: float, funding_rate: float,
                   index_price: float) -> float:
    """
    Calculate funding payment for a position.

    Formula: Payment = Position Size * Funding Rate * Index Price

    Positive payment means longs pay shorts.

    Args:
        position_size: Signed position size (positive=long, negative=short)
        funding_rate: Current funding rate
        index_price: Oracle index price

    Returns:
        Funding payment amount (positive = pay, negative = receive)
    """
    return position_size * funding_rate * index_price


# =============================================================================
# PNL CALCULATIONS
# =============================================================================

def unrealized_pnl(position_size: float, entry_price: float,
                   mark_price: float) -> float:
    """
    Calculate unrealized PnL.

    Formula: PnL = Position Size * (Mark Price - Entry Price)

    Args:
        position_size: Signed position size (positive=long, negative=short)
        entry_price: Average entry price
        mark_price: Current mark price

    Returns:
        Unrealized PnL in quote currency
    """
    return position_size * (mark_price - entry_price)


def equity(collateral: float, unrealized_pnl: float,
           accumulated_funding: float) -> float:
    """
    Calculate account equity.

    Formula: Equity = Collateral + Unrealized PnL - Accumulated Funding Paid

    Args:
        collateral: Deposited collateral
        unrealized_pnl: Current unrealized PnL
        accumulated_funding: Total funding payments made (positive = paid out)

    Returns:
        Account equity
    """
    return collateral + unrealized_pnl - accumulated_funding


# =============================================================================
# RISK LIMIT CALCULATIONS
# =============================================================================

def max_position_size(margin: float, leverage: float, price: float) -> float:
    """
    Calculate maximum position size for given margin and leverage.

    Formula: Max Size = (Margin * Leverage) / Price

    Args:
        margin: Available margin
        leverage: Desired leverage
        price: Current price

    Returns:
        Maximum position size in base asset
    """
    return (margin * leverage) / price if price > 0 else 0


def max_oi_from_liquidity(liquidity_usd: float, risk_factor: float = 0.2) -> float:
    """
    Calculate maximum OI based on available liquidity.

    Conservative approach: OI should not exceed a fraction of liquidity
    to ensure orderly liquidations.

    Args:
        liquidity_usd: Total orderbook liquidity in USD
        risk_factor: Fraction of liquidity for max OI (default 20%)

    Returns:
        Maximum recommended OI in USD
    """
    return liquidity_usd * risk_factor


def liquidation_risk_score(oi_usd: float, liquidity_usd: float,
                          volatility: float) -> Tuple[int, str]:
    """
    Calculate liquidation cascade risk score.

    Args:
        oi_usd: Current open interest
        liquidity_usd: Available liquidity
        volatility: Annualized volatility (0-1)

    Returns:
        Tuple of (score 1-5, description)
    """
    oi_to_liq = oi_usd / liquidity_usd if liquidity_usd > 0 else float('inf')
    vol_factor = 1 + volatility  # Higher vol = more risk

    adjusted_ratio = oi_to_liq * vol_factor

    if adjusted_ratio < 0.1:
        return 1, "Very Low Risk"
    elif adjusted_ratio < 0.2:
        return 2, "Low Risk"
    elif adjusted_ratio < 0.4:
        return 3, "Medium Risk"
    elif adjusted_ratio < 0.8:
        return 4, "High Risk"
    else:
        return 5, "Critical Risk"


# =============================================================================
# MANIPULATION ANALYSIS (for oracle/perp manipulation scenarios)
# =============================================================================

def manipulation_profit(position_size_usd: float, price_impact_pct: float,
                       manipulation_cost_usd: float,
                       funding_periods: int = 1,
                       funding_rate: float = 0.01) -> Dict:
    """
    Analyze profit/loss from price manipulation attack.

    Attacker strategy:
    1. Open leveraged position
    2. Manipulate spot/oracle price
    3. Profit from position PnL
    4. Close position

    Args:
        position_size_usd: Notional position size
        price_impact_pct: Price move achieved (%)
        manipulation_cost_usd: Cost to manipulate price
        funding_periods: Number of funding periods position is held
        funding_rate: Funding rate per period

    Returns:
        Dictionary with profit analysis
    """
    # Position PnL
    position_pnl = position_size_usd * (price_impact_pct / 100)

    # Funding cost
    funding_cost = position_size_usd * funding_rate * funding_periods

    # Liquidation fee if triggered (1%)
    liq_fee = position_size_usd * (LIQUIDATION_FEE_BPS / 10000)

    # Net profit
    gross_profit = position_pnl - manipulation_cost_usd
    net_profit = gross_profit - funding_cost

    return {
        "position_size_usd": position_size_usd,
        "price_impact_pct": price_impact_pct,
        "position_pnl": position_pnl,
        "manipulation_cost": manipulation_cost_usd,
        "funding_cost": funding_cost,
        "potential_liq_fee": liq_fee,
        "gross_profit": gross_profit,
        "net_profit": net_profit,
        "profitable": net_profit > 0,
        "breakeven_impact_pct": (manipulation_cost_usd / position_size_usd) * 100 if position_size_usd > 0 else float('inf')
    }


def safe_oi_limit(orderbook_depth_usd: float, max_leverage: float,
                  mm_factor: float, safety_margin: float = 2.0) -> float:
    """
    Calculate safe OI limit based on manipulation economics.

    Safe when: Manipulation Cost > Max Profit
    Max Profit = Position * Price Impact
    Safe OI = Depth / (Leverage * Price Impact) * Safety Margin

    Args:
        orderbook_depth_usd: Total orderbook depth to 1% impact
        max_leverage: Maximum allowed leverage
        mm_factor: Maintenance margin factor
        safety_margin: Additional safety multiplier

    Returns:
        Recommended maximum OI in USD
    """
    # At max leverage, small price moves can liquidate
    # We want manipulation cost > potential profit

    # Simplified: OI should be small enough that moving price
    # costs more than the profit from a max-leverage position

    # If depth = $10M to move 1%, and max leverage = 20x
    # Max profit from 1% move = Position * 1% = OI * 1%
    # Cost to achieve = $10M
    # Safe: $10M > OI * 1% => OI < $1B

    # But we also consider that attacker needs margin for position
    # More conservative: OI < Depth / Safety

    return orderbook_depth_usd / safety_margin


# Export all
__all__ = [
    # Constants
    'MAX_OPEN_ORDERS', 'HEALTH_FACTOR_LIQUIDATION_THRESHOLD',
    'CLOSEOUT_MARGIN_FACTOR', 'LIQUIDATION_FEE_BPS',
    'FUNDING_CAP', 'FUNDING_FLOOR', 'INTEREST_RATE',

    # Classes
    'MarketConfig', 'DEFAULT_MARKET_CONFIGS',

    # Margin functions
    'initial_margin', 'maintenance_margin',
    'is_liquidatable', 'is_closeout', 'liquidation_price',

    # Funding functions
    'premium_index', 'funding_rate', 'funding_payment',

    # PnL functions
    'unrealized_pnl', 'equity',

    # Risk functions
    'max_position_size', 'max_oi_from_liquidity',
    'liquidation_risk_score', 'manipulation_profit', 'safe_oi_limit',
]
