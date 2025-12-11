"""Plotting utilities for risk visualization"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.colors as mcolors

# Professional chart styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.edgecolor': '0.8',
    'figure.autolayout': True
})

# Professional color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#62C370',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6C757D',
    'dark': '#343A40',
    'light': '#F8F9FA'
}

# Gradient colormap for multi-line plots
def get_color_gradient(n_colors: int, cmap_name: str = 'viridis') -> List[str]:
    """Get evenly spaced colors from a colormap"""
    cmap = cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]

def plot_orderbook_depth(liq_curve: pd.DataFrame, title: str = "Orderbook Depth") -> Tuple[Figure, Axes]:
    """
    Plot orderbook depth curves
    
    Args:
        liq_curve: Liquidity curve DataFrame
        title: Plot title
        
    Returns:
        Figure and axes objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate bid and ask data
    bid_data = liq_curve[liq_curve['side'] == 'bid'].copy()
    ask_data = liq_curve[liq_curve['side'] == 'ask'].copy()
    
    # Plot cumulative notional vs price impact
    if not bid_data.empty:
        ax1.plot(bid_data['price_impact_pct'], bid_data['cum_notional']/1e6, 
                'r-', label='Bids', linewidth=2)
    if not ask_data.empty:
        ax1.plot(ask_data['price_impact_pct'], ask_data['cum_notional']/1e6,
                'g-', label='Asks', linewidth=2)
    
    ax1.set_xlabel('Price Impact (%)')
    ax1.set_ylabel('Cumulative Notional (USD millions)')
    ax1.set_title(f'{title} - Liquidity Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot price levels
    if not liq_curve.empty:
        mid_price = liq_curve.iloc[0]['mid_price']
        
        if not bid_data.empty:
            ax2.barh(bid_data['price'], bid_data['qty'], 
                    height=(bid_data['price'].diff().fillna(bid_data['price'].iloc[0]*0.001)), 
                    color='red', alpha=0.6, label='Bids')
        if not ask_data.empty:
            ax2.barh(ask_data['price'], ask_data['qty'],
                    height=(ask_data['price'].diff().fillna(ask_data['price'].iloc[0]*0.001)),
                    color='green', alpha=0.6, label='Asks')
        
        ax2.axhline(mid_price, color='black', linestyle='--', label='Mid Price')
        ax2.set_xlabel('Quantity')
        ax2.set_ylabel('Price')
        ax2.set_title(f'{title} - Order Book')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_volatility_history(klines_df: pd.DataFrame, window: int = 24*7) -> Tuple[Figure, Axes]:
    """
    Plot price and rolling volatility
    
    Args:
        klines_df: OHLC data
        window: Rolling window for vol calc
        
    Returns:
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price chart
    ax1.plot(klines_df.index, klines_df['close'], 'b-', linewidth=1)
    ax1.fill_between(klines_df.index, klines_df['low'], klines_df['high'], 
                     alpha=0.1, color='blue')
    ax1.set_ylabel('Price')
    ax1.set_title('Price History')
    ax1.grid(True, alpha=0.3)
    
    # Volatility chart
    returns = np.log(klines_df['close'] / klines_df['close'].shift(1))
    rolling_vol = returns.rolling(window).std() * np.sqrt(365 * 24) * 100  # Annualized %
    
    ax2.plot(klines_df.index, rolling_vol, 'r-', linewidth=1)
    ax2.fill_between(klines_df.index, 0, rolling_vol, alpha=0.3, color='red')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title(f'{window}-hour Rolling Volatility (Annualized)')
    ax2.grid(True, alpha=0.3)
    
    # Add volatility bands
    for vol, label in [(30, 'Low'), (50, 'Medium'), (80, 'High')]:
        ax2.axhline(vol, color='gray', linestyle='--', alpha=0.5)
        ax2.text(klines_df.index[-1], vol, label, ha='left', va='bottom')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_risk_scores(scores_df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """
    Plot risk scores comparison
    
    Args:
        scores_df: DataFrame with columns for different risk scores
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(scores_df, annot=True, fmt='d', cmap='RdYlGn_r',
                center=3, vmin=1, vmax=5, cbar_kws={'label': 'Risk Score'},
                ax=ax)
    
    ax.set_title('Risk Score Matrix')
    ax.set_xlabel('Risk Factor')
    ax.set_ylabel('Asset')
    
    plt.tight_layout()
    return fig, ax

def plot_liquidity_comparison(markets_data: Dict[str, Dict]) -> Tuple[Figure, Axes]:
    """
    Compare liquidity across markets
    
    Args:
        markets_data: Dictionary of market data by symbol
        
    Returns:
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    symbols = []
    cex_liq = []
    amm_liq = []
    total_liq = []
    
    for symbol, data in markets_data.items():
        symbols.append(symbol)
        cex_liq.append(data.get('cex_liquidity_usd', 0) / 1e6)
        amm_liq.append(data.get('amm_adjusted_usd', 0) / 1e6)
        total_liq.append(data.get('total_liquidity_usd', 0) / 1e6)
    
    # Stacked bar chart
    x = np.arange(len(symbols))
    width = 0.35
    
    ax1.bar(x, cex_liq, width, label='CEX', color='blue', alpha=0.8)
    ax1.bar(x, amm_liq, width, bottom=cex_liq, label='AMM (adjusted)', 
            color='green', alpha=0.8)
    
    ax1.set_ylabel('Liquidity (USD millions)')
    ax1.set_title('Liquidity Sources by Market')
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart of total liquidity
    ax2.pie(total_liq, labels=symbols, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Total Liquidity Distribution')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_scenario_impacts(scenarios_df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """
    Plot stress scenario impacts
    
    Args:
        scenarios_df: DataFrame with scenario results
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by scenario type
    scenario_types = scenarios_df['type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_types)))
    
    # Create grouped bar chart
    scenarios_df['impact_pct'] = scenarios_df['impact_pct'].fillna(0)
    pivoted = scenarios_df.pivot(index='scenario', columns='type', values='impact_pct')
    
    pivoted.plot(kind='bar', ax=ax, color=colors, width=0.8)
    
    # Add severity coloring
    for i, (idx, row) in enumerate(scenarios_df.iterrows()):
        severity = row['severity']
        if severity == 'high':
            ax.axhspan(i-0.4, i+0.4, alpha=0.1, color='red')
        elif severity == 'medium':
            ax.axhspan(i-0.4, i+0.4, alpha=0.1, color='yellow')
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Impact (%)')
    ax.set_title('Stress Scenario Impacts')
    ax.legend(title='Scenario Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, ax

def plot_volume_depth_correlation(volume_data: Dict[str, pd.DataFrame]) -> Tuple[Figure, Axes]:
    """
    Plot correlation between trading volume and market depth
    
    Args:
        volume_data: Dictionary with volume and depth time series
        
    Returns:
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Volume over time
    if 'volume' in volume_data:
        vol_df = volume_data['volume']
        ax1.fill_between(vol_df.index, 0, vol_df['volume'] / 1e6, 
                        alpha=0.3, color=COLORS['primary'])
        ax1.plot(vol_df.index, vol_df['volume'] / 1e6, 
                color=COLORS['primary'], linewidth=2)
        ax1.set_ylabel('24h Volume (USD millions)', fontsize=12)
        ax1.set_title('Trading Volume and Market Depth Analysis', fontsize=14, fontweight='bold')
        
    # Depth over time
    if 'depth' in volume_data:
        depth_df = volume_data['depth']
        ax2.plot(depth_df.index, depth_df['depth_1pct'] / 1e6,
                label='1% depth', color=COLORS['success'], linewidth=2)
        ax2.plot(depth_df.index, depth_df['depth_2pct'] / 1e6,
                label='2% depth', color=COLORS['warning'], linewidth=2)
        ax2.fill_between(depth_df.index, 0, depth_df['depth_1pct'] / 1e6,
                        alpha=0.2, color=COLORS['success'])
        ax2.set_ylabel('Market Depth (USD millions)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper right')
        
    # Add correlation text
    if 'volume' in volume_data and 'depth' in volume_data:
        corr = vol_df['volume'].corr(depth_df['depth_1pct'])
        ax2.text(0.02, 0.98, f'Volume-Depth Correlation: {corr:.3f}',
                transform=ax2.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_risk_heatmap(risk_matrix: pd.DataFrame, title: str = "Risk Score Heatmap") -> Tuple[Figure, Axes]:
    """
    Create a professional heatmap for risk scores
    
    Args:
        risk_matrix: DataFrame with risk scores
        title: Plot title
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap (green to red)
    colors = ['#62C370', '#F9DC5C', '#F18F01', '#C73E1D', '#8B0000']
    n_bins = 5
    cmap = mcolors.LinearSegmentedColormap.from_list('risk', colors, N=n_bins)
    
    # Plot heatmap
    sns.heatmap(risk_matrix, annot=True, fmt='.2f', cmap=cmap,
                vmin=1, vmax=5, center=3,
                cbar_kws={'label': 'Risk Score', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray',
                square=True, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risk Factor', fontsize=12)
    ax.set_ylabel('Asset', fontsize=12)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig, ax

def plot_volatility_surface(vol_data: Dict[str, pd.DataFrame], 
                          title: str = "Volatility Surface") -> Tuple[Figure, Axes]:
    """
    Plot volatility surface across different time horizons
    
    Args:
        vol_data: Dictionary with volatility data by asset
        title: Plot title
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get color gradient
    assets = list(vol_data.keys())
    colors = get_color_gradient(len(assets), 'rainbow')
    
    for i, (asset, df) in enumerate(vol_data.items()):
        if 'volatility' in df.columns:
            ax.plot(df.index, df['volatility'] * 100, 
                   label=asset, color=colors[i], linewidth=2.5)
            
            # Add confidence bands
            if 'vol_lower' in df.columns and 'vol_upper' in df.columns:
                ax.fill_between(df.index, 
                              df['vol_lower'] * 100, 
                              df['vol_upper'] * 100,
                              alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Add volatility regime bands
    regime_levels = [30, 50, 80, 120]
    regime_labels = ['Low', 'Medium', 'High', 'Extreme']
    for level, label in zip(regime_levels, regime_labels):
        ax.axhline(level, color='gray', linestyle='--', alpha=0.5)
        ax.text(df.index[-1], level, label, ha='left', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig, ax

def plot_pnl_distribution(pnl_scenarios: pd.DataFrame,
                         title: str = "P&L Distribution") -> Tuple[Figure, Axes]:
    """
    Plot P&L distribution with percentiles
    
    Args:
        pnl_scenarios: DataFrame with P&L scenarios
        title: Plot title
        
    Returns:
        Figure and axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # PDF plot
    for col in pnl_scenarios.columns:
        if 'pnl' in col.lower():
            data = pnl_scenarios[col].dropna()
            
            # Calculate histogram
            hist, bins = np.histogram(data, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Smooth with gaussian kernel
            from scipy.ndimage import gaussian_filter1d
            hist_smooth = gaussian_filter1d(hist, sigma=1.5)
            
            ax1.plot(bin_centers, hist_smooth, linewidth=2.5, label=col)
            ax1.fill_between(bin_centers, 0, hist_smooth, alpha=0.2)
    
    ax1.set_xlabel('P&L (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('P&L Probability Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = np.percentile(data, p)
        ax1.axvline(val, color='red', linestyle='--', alpha=0.5)
        ax1.text(val, ax1.get_ylim()[1]*0.9, f'{p}%', rotation=90, va='top')
    
    # CDF plot
    for col in pnl_scenarios.columns:
        if 'pnl' in col.lower():
            data = sorted(pnl_scenarios[col].dropna())
            y = np.arange(1, len(data) + 1) / len(data)
            ax2.plot(data, y, linewidth=2.5, label=col)
    
    ax2.set_xlabel('P&L (%)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('P&L Cumulative Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add risk metrics
    var_95 = np.percentile(data, 5)
    cvar_95 = np.mean([x for x in data if x <= var_95])
    ax2.axvline(var_95, color='red', linestyle='--', linewidth=2)
    ax2.text(var_95, 0.5, f'VaR 95%: {var_95:.1f}%', rotation=90, va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_oi_limits_table(limits_df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """
    Create a formatted table of OI limits
    
    Args:
        limits_df: DataFrame with OI limits by market
        
    Returns:
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format numbers
    for col in ['max_oi_usd', 'max_position_usd', 'total_liquidity_usd']:
        if col in limits_df.columns:
            limits_df[col] = limits_df[col].apply(lambda x: f'${x/1e6:.1f}M')
    
    # Create table
    table = ax.table(cellText=limits_df.values,
                     colLabels=limits_df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code by risk score
    if 'risk_score' in limits_df.columns:
        risk_col_idx = limits_df.columns.get_loc('risk_score')
        for i, risk_score in enumerate(limits_df['risk_score']):
            if risk_score >= 4:
                color = '#ffcccc'  # Light red
            elif risk_score >= 3:
                color = '#ffffcc'  # Light yellow
            else:
                color = '#ccffcc'  # Light green
            
            for j in range(len(limits_df.columns)):
                table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('Recommended OI Limits by Market', fontsize=14, fontweight='bold', pad=20)
    
    return fig, ax

def plot_risk_limits_comparison(risk_metrics: pd.DataFrame, 
                              figsize: Tuple[float, float] = (16, 10)) -> Tuple[Figure, Axes]:
    """
    Create comprehensive risk limits comparison visualization
    
    Args:
        risk_metrics: DataFrame with risk metrics and limits
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Sort by risk score for better visualization
    risk_sorted = risk_metrics.sort_values('risk_score')
    top_markets = risk_sorted.head(10)
    
    # 1. Risk Score vs Position Limits
    scatter = ax1.scatter(top_markets['risk_score'], 
                         top_markets['position_limit_mm'],
                         c=top_markets['liquidity_1pct_mm'],
                         s=200, cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Add market labels
    for idx, row in top_markets.iterrows():
        ax1.annotate(row['market'], 
                    (row['risk_score'], row['position_limit_mm']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Risk Score (1-5)')
    ax1.set_ylabel('Position Limit ($MM)')
    ax1.set_title('Risk Score vs Position Limits')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for liquidity
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Liquidity @1% ($MM)')
    
    # 2. Liquidity vs Max OI
    ax2.scatter(top_markets['liquidity_1pct_mm'],
               top_markets['max_oi_mm'],
               s=100, alpha=0.6, color=COLORS['primary'])
    
    # Add trend line
    z = np.polyfit(top_markets['liquidity_1pct_mm'], top_markets['max_oi_mm'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(top_markets['liquidity_1pct_mm'].min(), 
                         top_markets['liquidity_1pct_mm'].max(), 100)
    ax2.plot(x_trend, p(x_trend), '--', color=COLORS['secondary'], alpha=0.8)
    
    ax2.set_xlabel('Liquidity @1% Impact ($MM)')
    ax2.set_ylabel('Max Open Interest ($MM)')
    ax2.set_title('Liquidity vs Maximum OI')
    ax2.grid(True, alpha=0.3)
    
    # 3. Position Limits by Market (Bar chart)
    markets_display = top_markets['market'].tolist()
    y_pos = np.arange(len(markets_display))
    
    ax3.barh(y_pos, top_markets['position_limit_mm'], 
             color=[COLORS['success'] if score <= 2 else 
                   COLORS['warning'] if score <= 3 else 
                   COLORS['danger'] 
                   for score in top_markets['risk_score']])
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(markets_display)
    ax3.set_xlabel('Position Limit ($MM)')
    ax3.set_title('Position Limits by Market')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Risk Metrics Heatmap
    # Create subset of metrics for heatmap
    heatmap_data = top_markets[['market', 'risk_score', 'liquidity_score', 
                               'volatility_score', 'oracle_score']].set_index('market')
    
    # Normalize scores to 0-1 for better color mapping
    heatmap_norm = (heatmap_data - 1) / 4  # Since scores are 1-5
    
    sns.heatmap(heatmap_norm.T, 
                annot=heatmap_data.T,  # Show actual values
                fmt='.0f',
                cmap='RdYlGn_r',  # Reversed so red=bad, green=good
                cbar_kws={'label': 'Risk Level'},
                vmin=0, vmax=1,
                ax=ax4)
    
    ax4.set_title('Risk Component Breakdown')
    ax4.set_xlabel('Market')
    
    plt.suptitle('Market Risk Limits Analysis', fontsize=18, y=1.02)
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3, ax4)