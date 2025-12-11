"""Scenario configurations for risk analysis notebooks"""

# Attack scenario configurations
MANIPULATION_SCENARIOS = {
    'Simple Pump': {
        'spot_manipulation': 5,
        'leverage': 10,
        'position_size': 1e6,
        'funding_game': False
    },
    'Cascade Hunt': {
        'spot_manipulation': 10,
        'leverage': 25,
        'position_size': 5e6,
        'funding_game': False
    },
    'Funding Squeeze': {
        'spot_manipulation': 2,
        'leverage': 5,
        'position_size': 10e6,
        'funding_game': True
    },
    'Full Attack': {
        'spot_manipulation': 10,
        'leverage': 50,
        'position_size': 10e6,
        'funding_game': True
    }
}

# Stress test scenario configurations
COMPOSITE_STRESS_SCENARIOS = {
    'Market Crash': {
        'liquidation_pct': 0.30,
        'gap_size': 0.10,
        'liquidity_reduction': 0.70,
        'volatility_multiplier': 3.0
    },
    'Death Spiral': {
        'liquidation_pct': 0.50,
        'gap_size': 0.20,
        'liquidity_reduction': 0.90,
        'volatility_multiplier': 5.0
    },
    'Black Thursday': {
        'liquidation_pct': 0.40,
        'gap_size': 0.15,
        'liquidity_reduction': 0.80,
        'volatility_multiplier': 4.0
    }
}

# Funding scenario configurations
FUNDING_SCENARIOS = {
    'normal': {'rate': 0.01, 'duration': 1},      # 0.01% for 1 period
    'elevated': {'rate': 0.1, 'duration': 3},     # 0.1% for 3 periods  
    'extreme': {'rate': 0.5, 'duration': 8},      # 0.5% for 8 periods
    'crisis': {'rate': 1.0, 'duration': 24}       # 1.0% for 24 periods
}

# Risk tier configurations
PARTICIPANT_TIERS = {
    'Retail': 0.05,         # 5% of calculated limit
    'Professional': 0.20,   # 20% of calculated limit
    'Market Maker': 0.50,   # 50% of calculated limit
    'Institutional': 1.0    # 100% of calculated limit
}

# Defense parameter recommendations
DEFENSE_PARAMETERS = {
    'Oracle Configuration': {
        'Min Venues': 5,
        'TWAP Window': '5 minutes',
        'Outlier Threshold': '3 sigma',
        'Circuit Breaker': '10%'
    },
    'Position Controls': {
        'Max Leverage': 50,
        'Position Limit': '25% of 1% depth',
        'OI Cap': '50% of daily volume',
        'Concentration Limit': '10% of total OI'
    },
    'Funding Controls': {
        'Max Rate': '0.5% per period',
        'Smoothing Period': '1 hour',
        'Alert Threshold': '0.1% change',
        'Emergency Cap': '1% per period'
    },
    'Economic Incentives': {
        'Maker Rebate': '0.02%',
        'Taker Fee': '0.05%',
        'Depth Bonus': '0.01% for >$1M',
        'Stability Fund': '10% of fees'
    }
}

# Sample spot venues for oracle analysis
SAMPLE_SPOT_VENUES = [
    {'name': 'Binance', 'liquidity_usd': 10e6},
    {'name': 'Coinbase', 'liquidity_usd': 8e6},
    {'name': 'Kraken', 'liquidity_usd': 5e6},
    {'name': 'OKX', 'liquidity_usd': 7e6},
    {'name': 'Uniswap', 'liquidity_usd': 3e6}
]

# Analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    'liquidation_scenarios': [0.1, 0.2, 0.4],  # 10%, 20%, 40% of OI
    'gap_scenarios': [0.05, 0.10, 0.20],       # 5%, 10%, 20% gaps
    'volatility_shocks': [2, 3, 5],            # 2x, 3x, 5x multipliers
    'manipulation_impacts': [0.5, 1, 2, 5, 10, 20],  # Price impact percentages
    'leverage_levels': [5, 10, 25, 50],
    'position_sizes': [100000, 500000, 1000000, 5000000],
    'imbalance_sizes': [1e6, 5e6, 10e6, 25e6, 50e6]
}

# Chart configuration defaults
CHART_DEFAULTS = {
    'figure_size': (12, 8),
    'figure_size_wide': (14, 8),
    'figure_size_square': (10, 8),
    'figure_size_tall': (10, 10),
    'bar_width': 0.35,
    'alpha_fill': 0.3,
    'grid_alpha': 0.3
}

# Alert thresholds
ALERT_THRESHOLDS = {
    'utilization': {
        'warning': 50,
        'critical': 75,
        'maximum': 90
    },
    'volatility': {
        'normal': 0.5,    # 50% annualized
        'elevated': 1.0,  # 100% annualized
        'extreme': 2.0    # 200% annualized
    },
    'liquidity': {
        'deep': 10e6,     # $10MM at 1% 
        'moderate': 5e6,  # $5MM at 1%
        'thin': 1e6       # $1MM at 1%
    }
}

# Export all configurations
__all__ = [
    'MANIPULATION_SCENARIOS',
    'COMPOSITE_STRESS_SCENARIOS',
    'FUNDING_SCENARIOS',
    'PARTICIPANT_TIERS',
    'DEFENSE_PARAMETERS',
    'SAMPLE_SPOT_VENUES',
    'DEFAULT_ANALYSIS_PARAMS',
    'CHART_DEFAULTS',
    'ALERT_THRESHOLDS'
]