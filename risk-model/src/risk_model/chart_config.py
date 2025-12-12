"""Chart configuration and styling for notebooks"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Professional chart styling configuration
CHART_STYLE = {
    'figure.figsize': (12, 6),
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.edgecolor': '0.8',
    'legend.borderaxespad': 0.5,
    'figure.autolayout': True,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}

# Line styles for profit analysis charts
LINE_STYLES = ['-', '--', '-.', ':', '-']
MARKERS = ['o', 's', '^', 'D', 'v']

# Color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#62C370',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6C757D',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'background': '#FFFFFF',
    'grid': '#E0E0E0'
}

# Risk color mapping
RISK_COLORS = {
    1: '#62C370',  # Green - Low risk
    2: '#8BC34A',  # Light green
    3: '#F9DC5C',  # Yellow - Medium risk
    4: '#F18F01',  # Orange
    5: '#C73E1D'   # Red - High risk
}

# Market-specific colors for consistency
MARKET_COLORS = {
    'ETH-PERP': '#627EEA',
    'BTC-PERP': '#F7931A',
    'SOL-PERP': '#14F195',
    'ARB-PERP': '#28A0F0',
    'MATIC-PERP': '#8247E5',
    'DEFAULT': '#6C757D'
}

def setup_chart_style():
    """Apply professional chart styling"""
    plt.style.use('seaborn-v0_8-whitegrid')
    rcParams.update(CHART_STYLE)
    
    # Set seaborn context for better scaling
    sns.set_context("notebook", font_scale=1.1)
    
    # Update seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '.15',
        'axes.grid': True,
        'grid.color': '.9',
        'grid.linestyle': '-',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

def get_market_color(market_name: str) -> str:
    """Get consistent color for a market"""
    return MARKET_COLORS.get(market_name, MARKET_COLORS['DEFAULT'])

def get_risk_color(risk_score: int) -> str:
    """Get color based on risk score"""
    risk_score = max(1, min(5, int(risk_score)))  # Clamp to 1-5
    return RISK_COLORS.get(risk_score, COLORS['info'])

def format_axis_labels(ax, xlabel: str = None, ylabel: str = None, 
                      title: str = None, grid: bool = True):
    """Apply consistent axis formatting"""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def add_value_labels(ax, bars, format_string='{:.0f}', offset=0.01):
    """Add value labels to bar charts"""
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label positive values
            ax.annotate(format_string.format(height),
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)

def create_figure_with_title(title: str, figsize: tuple = (12, 8)):
    """Create a figure with consistent styling and title"""
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    return fig

# Chart templates for common visualizations
CHART_TEMPLATES = {
    'risk_matrix': {
        'figsize': (10, 8),
        'cmap': 'RdYlGn_r',
        'vmin': 1,
        'vmax': 5,
        'center': 3,
        'linewidths': 0.5,
        'linecolor': 'gray',
        'square': True,
        'cbar_kws': {'label': 'Risk Score', 'shrink': 0.8},
        'annot_kws': {'fontsize': 11}
    },
    'depth_chart': {
        'figsize': (14, 8),
        'bid_color': COLORS['danger'],
        'ask_color': COLORS['success'],
        'alpha': 0.7,
        'linewidth': 2.5
    },
    'volatility_chart': {
        'figsize': (14, 8),
        'color_palette': 'viridis',
        'alpha_fill': 0.2,
        'linewidth': 2.5
    },
    'volume_chart': {
        'figsize': (14, 10),
        'volume_color': COLORS['primary'],
        'depth_colors': [COLORS['success'], COLORS['warning']],
        'alpha': 0.3
    }
}

# Synthetic orderbook generation
def create_synthetic_orderbook(
    mid_price: float = 100.0,
    depth_per_level_usd: float = 10_000,
    num_levels: int = 100,
    depth_decay: float = 0.95,
    spread_bps: float = 10
) -> dict:
    """Create a realistic synthetic orderbook."""
    bids, asks = [], []
    half_spread = mid_price * spread_bps / 20000
    
    for i in range(num_levels):
        # Bid side
        bid_price = (mid_price - half_spread) * (1 - 0.001 * i)
        bid_depth = depth_per_level_usd * (depth_decay ** i)
        bids.append([bid_price, bid_depth / bid_price])
        
        # Ask side
        ask_price = (mid_price + half_spread) * (1 + 0.001 * i)
        ask_depth = depth_per_level_usd * (depth_decay ** i)
        asks.append([ask_price, ask_depth / ask_price])
    
    return {'bids': bids, 'asks': asks, 'mid_price': mid_price}

# Manipulation calculations
def calculate_manipulation_cost(orderbook: dict, target_impact_pct: float) -> float:
    """Calculate cost to move price by target percentage."""
    mid_price = orderbook['mid_price']
    target_price = mid_price * (1 + target_impact_pct / 100)
    
    total_cost = 0
    for price, size in orderbook['asks']:
        if price > target_price:
            break
        total_cost += price * size
    
    return total_cost

def calculate_net_profit(manipulation_cost: float, position_size: float, 
                        price_impact_pct: float, leverage: float = 20) -> float:
    """Calculate attacker's net profit."""
    position_pnl = position_size * (price_impact_pct / 100)
    funding_cost = position_size * 0.0001  # 0.01% funding
    return position_pnl - manipulation_cost - funding_cost

# Safe OI calculation functions
def find_breakeven_position(manipulation_cost: float, price_impact_pct: float) -> float:
    """Find position size where profit = 0."""
    return manipulation_cost / (price_impact_pct / 100)

def calculate_safe_oi(orderbook: dict, target_impact: float, 
                     safety_factor: float = 2.0) -> float:
    """Calculate safe OI limit."""
    manip_cost = calculate_manipulation_cost(orderbook, target_impact)
    breakeven = find_breakeven_position(manip_cost, target_impact)
    return breakeven / safety_factor

# Heatmap configuration
HEATMAP_CONFIG = {
    'figsize': (12, 8),
    'cmap': 'RdYlGn',
    'interpolation': 'bilinear',
    'profit_levels': [-2, -1, 1, 2, 5],
    'contour_colors': {
        'breakeven': 'black',
        'profit': 'white'
    },
    'annotations': {
        'profitable': {
            'position': (10, 40),  # top left
            'text': 'PROFITABLE\nZONE',
            'fontsize': 12
        },
        'unprofitable': {
            'position': (40, 10),  # bottom right
            'text': 'UNPROFITABLE\nZONE',
            'fontsize': 12
        }
    }
}

# Export all configuration
__all__ = [
    'CHART_STYLE',
    'COLORS', 
    'RISK_COLORS',
    'MARKET_COLORS',
    'CHART_TEMPLATES',
    'LINE_STYLES',
    'MARKERS',
    'HEATMAP_CONFIG',
    'setup_chart_style',
    'get_market_color',
    'get_risk_color',
    'format_axis_labels',
    'add_value_labels',
    'create_figure_with_title',
    'create_synthetic_orderbook',
    'calculate_manipulation_cost',
    'calculate_net_profit',
    'find_breakeven_position',
    'calculate_safe_oi'
]