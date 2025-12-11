# CLOB Risk Modelling Framework

Risk modelling framework for Perpetual CLOBs, combining CEX (Binance) and AMM (Uniswap) liquidity analysis.

## Overview

This framework provides tools to:
- Analyze orderbook depth and liquidity from Binance
- Assess AMM liquidity from Uniswap v3 pools
- Calculate volatility metrics and risk scores
- Recommend maximum OI and position limits
- Run stress test scenarios

## Quick Start

1. **Install dependencies:**
   ```bash
   make setup
   ```

2. **Launch Jupyter notebooks:**
   ```bash
   make notebook
   ```

3. **Run data exploration:**
   Open `00_data_exploration.ipynb` in Jupyter

## Project Structure

```
risk-model/
├── notebooks/          # Jupyter notebooks for analysis
├── src/risk_model/     # Core Python modules
│   ├── binance_data.py # Binance API integration
│   ├── uniswap_data.py # Uniswap subgraph queries
│   ├── pricing.py      # Liquidity and volatility analysis
│   ├── metrics.py      # Risk scoring and limits
│   ├── scenarios.py    # Stress testing
│   └── plotting.py     # Visualization utilities
└── config/
    └── markets.yaml    # Market configurations
```

## Key Features

### 1. Data Collection
- **Binance**: Orderbook snapshots, historical OHLC, 24h volume
- **Uniswap**: Pool TVL, liquidity distribution, price impact estimates

### 2. Risk Metrics
- **Liquidity Score**: Based on depth at 1% price impact
- **Volatility Score**: Using realized volatility
- **Oracle Score**: Quality of price feeds
- **Composite Risk Score**: Weighted combination

### 3. Position Limits
- Maximum OI as percentage of available liquidity
- Single position limits
- OI to volume ratio monitoring

### 4. Stress Testing
- Mass liquidation scenarios
- Price gap risk analysis
- Volatility shock impacts
- Cascading liquidation models

## Available Commands

```bash
make help         # Show all commands
make setup        # Install dependencies
make notebook     # Launch Jupyter
make fetch-data   # Fetch latest market data
make reports      # Generate HTML reports
make verify       # Verify installation
```

## Configuration

Edit `risk-model/config/markets.yaml` to:
- Add new markets
- Adjust risk parameters
- Configure API endpoints

## Next Steps

1. Complete market risk limits notebook (01)
2. Build stress testing notebook (02)
3. Add more sophisticated volatility models
4. Implement automated reporting
5. Add real-time monitoring capabilities