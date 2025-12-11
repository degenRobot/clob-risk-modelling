"""Configuration loader for CLOB risk modelling"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "markets.yaml"

def load_markets(config_path: Optional[str] = None) -> Dict:
    """
    Load markets configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default
        
    Returns:
        Dictionary with markets configuration
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded {len(config['markets'])} markets from {path}")
    return config

def get_market_config(market_name: str, config_path: Optional[str] = None) -> Dict:
    """
    Get configuration for a specific market
    
    Args:
        market_name: Name of the market (e.g., 'ETH-PERP')
        config_path: Path to config file
        
    Returns:
        Market configuration dictionary
    """
    config = load_markets(config_path)
    
    for market in config['markets']:
        if market['name'] == market_name:
            return market
            
    raise ValueError(f"Market {market_name} not found in configuration")

def get_risk_parameters(config_path: Optional[str] = None) -> Dict:
    """Get risk parameters from configuration"""
    config = load_markets(config_path)
    return config.get('risk_parameters', {})

def get_api_config(config_path: Optional[str] = None) -> Dict:
    """Get API configuration"""
    config = load_markets(config_path)
    return config.get('api_config', {})

def list_markets(config_path: Optional[str] = None) -> List[str]:
    """List all configured market names"""
    config = load_markets(config_path)
    return [market['name'] for market in config['markets']]

def get_markets_by_category(category: str, config_path: Optional[str] = None) -> List[Dict]:
    """Get all markets in a specific category"""
    config = load_markets(config_path)
    return [market for market in config['markets'] if market.get('category') == category]