"""Uniswap data fetching module for AMM liquidity analysis"""

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class UniswapDataFetcher:
    """Fetches pool data from Uniswap subgraphs"""
    
    def __init__(self, subgraph_url: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"):
        self.subgraph_url = subgraph_url
        transport = RequestsHTTPTransport(url=subgraph_url)
        # Don't fetch schema due to deprecated endpoint
        self.client = Client(transport=transport, fetch_schema_from_transport=False)
        
    def query_pool(self, pool_id: str) -> Dict:
        """
        Query Uniswap v3 pool data
        
        Args:
            pool_id: Pool address (lowercase, with 0x prefix)
            
        Returns:
            Dictionary with pool state data
        """
        query = gql("""
            query GetPool($poolId: String!) {
                pool(id: $poolId) {
                    id
                    token0 {
                        id
                        symbol
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        decimals
                    }
                    feeTier
                    liquidity
                    sqrtPrice
                    tick
                    token0Price
                    token1Price
                    totalValueLockedUSD
                    totalValueLockedToken0
                    totalValueLockedToken1
                    volumeUSD
                    volumeToken0
                    volumeToken1
                    txCount
                    poolDayData(first: 7, orderBy: date, orderDirection: desc) {
                        date
                        volumeUSD
                        tvlUSD
                        feesUSD
                        open
                        high
                        low
                        close
                    }
                }
            }
        """)
        
        try:
            result = self.client.execute(query, variable_values={"poolId": pool_id.lower()})
            return result.get("pool", {})
        except Exception as e:
            logger.error(f"Failed to query pool {pool_id}: {e}")
            raise
            
    def query_pool_hour_data(self, pool_id: str, hours: int = 24) -> List[Dict]:
        """
        Get hourly data for a pool
        
        Args:
            pool_id: Pool address
            hours: Number of hours to fetch
            
        Returns:
            List of hourly data dictionaries
        """
        query = gql("""
            query GetPoolHourData($poolId: String!, $hours: Int!) {
                poolHourDatas(
                    where: {pool: $poolId}
                    first: $hours
                    orderBy: periodStartUnix
                    orderDirection: desc
                ) {
                    periodStartUnix
                    liquidity
                    sqrtPrice
                    token0Price
                    token1Price
                    tick
                    volumeUSD
                    volumeToken0
                    volumeToken1
                    tvlUSD
                    feesUSD
                    txCount
                    open
                    high
                    low
                    close
                }
            }
        """)
        
        try:
            result = self.client.execute(
                query, 
                variable_values={"poolId": pool_id.lower(), "hours": hours}
            )
            return result.get("poolHourDatas", [])
        except Exception as e:
            logger.error(f"Failed to query hourly data for pool {pool_id}: {e}")
            raise
            
    def query_ticks(self, pool_id: str, num_ticks: int = 100) -> List[Dict]:
        """
        Get tick data for liquidity distribution
        
        Args:
            pool_id: Pool address
            num_ticks: Number of ticks to fetch around current price
            
        Returns:
            List of tick data
        """
        query = gql("""
            query GetTicks($poolId: String!, $numTicks: Int!) {
                ticks(
                    where: {pool: $poolId}
                    first: $numTicks
                    orderBy: tickIdx
                    orderDirection: asc
                ) {
                    tickIdx
                    liquidityGross
                    liquidityNet
                    price0
                    price1
                }
            }
        """)
        
        try:
            result = self.client.execute(
                query,
                variable_values={"poolId": pool_id.lower(), "numTicks": num_ticks}
            )
            return result.get("ticks", [])
        except Exception as e:
            logger.error(f"Failed to query ticks for pool {pool_id}: {e}")
            raise

def estimate_slippage_for_size(pool_state: Dict, trade_notional_usd: float, 
                              direction: str = "buy") -> Dict[str, float]:
    """
    Estimate price impact for a trade of given size
    
    This is a simplified approximation. For production use, implement
    proper Uniswap v3 math with tick traversal.
    
    Args:
        pool_state: Pool data from query_pool
        trade_notional_usd: Size of trade in USD
        direction: "buy" or "sell"
        
    Returns:
        Dictionary with estimated price impact metrics
    """
    try:
        tvl_usd = float(pool_state.get("totalValueLockedUSD", 0))
        fee_tier = int(pool_state.get("feeTier", 3000)) / 1e6  # Convert to decimal
        
        if tvl_usd == 0:
            return {"price_impact_pct": float('inf'), "effective_price_change_pct": float('inf')}
            
        # Simplified constant product approximation
        # Real v3 math would traverse ticks and calculate exact impact
        liquidity_consumed_pct = trade_notional_usd / tvl_usd
        
        # Rough approximation: price impact ~ liquidity consumed
        # In reality, concentrated liquidity makes this non-linear
        price_impact_pct = liquidity_consumed_pct * 100
        
        # Add fee impact
        fee_impact_pct = fee_tier * 100
        
        # Total effective price change
        effective_price_change_pct = price_impact_pct + fee_impact_pct
        
        return {
            "price_impact_pct": price_impact_pct,
            "fee_impact_pct": fee_impact_pct,
            "effective_price_change_pct": effective_price_change_pct,
            "liquidity_consumed_pct": liquidity_consumed_pct * 100
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate slippage: {e}")
        return {"price_impact_pct": float('inf'), "effective_price_change_pct": float('inf')}

def get_pool_liquidity_depth(pool_id: str, max_price_impact_pct: float = 1.0) -> Dict[str, float]:
    """
    Estimate available liquidity at given max price impact
    
    Args:
        pool_id: Pool address
        max_price_impact_pct: Maximum acceptable price impact
        
    Returns:
        Dictionary with liquidity metrics
    """
    fetcher = UniswapDataFetcher()
    pool_state = fetcher.query_pool(pool_id)
    
    if not pool_state:
        return {"max_trade_size_usd": 0, "tvl_usd": 0}
        
    tvl_usd = float(pool_state.get("totalValueLockedUSD", 0))
    fee_tier = int(pool_state.get("feeTier", 3000)) / 1e6
    
    # Reverse engineer max trade size from max impact
    # This is simplified - real implementation needs tick math
    fee_adjusted_impact = max_price_impact_pct - (fee_tier * 100)
    if fee_adjusted_impact <= 0:
        return {"max_trade_size_usd": 0, "tvl_usd": tvl_usd}
        
    # Approximate max trade size
    max_trade_size_usd = tvl_usd * (fee_adjusted_impact / 100)
    
    return {
        "max_trade_size_usd": max_trade_size_usd,
        "tvl_usd": tvl_usd,
        "fee_tier_pct": fee_tier * 100,
        "pool_id": pool_id
    }

# Convenience functions
def query_uniswap_v3_pool(pool_id: str) -> Dict:
    """Convenience function to query pool data"""
    fetcher = UniswapDataFetcher()
    return fetcher.query_pool(pool_id)

def get_pool_24h_volume(pool_id: str) -> float:
    """Get 24-hour volume for a pool"""
    fetcher = UniswapDataFetcher()
    hour_data = fetcher.query_pool_hour_data(pool_id, 24)
    
    if not hour_data:
        return 0.0
        
    total_volume = sum(float(h.get("volumeUSD", 0)) for h in hour_data)
    return total_volume