# CLOB Risk Modelling Framework

Comprehensive risk modelling framework for Perpetual CLOBs, analyzing market depth, volatility, and manipulation risks across centralized and decentralized venues.

## Overview

This framework provides enterprise-grade tools for:
- **Market Analysis**: Real-time orderbook depth and liquidity assessment
- **Risk Scoring**: Multi-factor risk evaluation with professional visualizations
- **Position Limits**: Dynamic limit calculation based on market conditions
- **Stress Testing**: Comprehensive scenario analysis including cascades and gaps
- **Manipulation Detection**: Economic analysis of attack vectors and defenses
- **Interactive Dashboards**: Real-time parameter adjustment and monitoring

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Run all notebooks with current data
make run-notebooks

# 3. Or launch interactive Jupyter environment
make notebook
```

## Key Notebooks

1. **Executive Summary** (`risk_summary.ipynb`) - High-level dashboard
2. **Data Exploration** (`00_data_exploration.ipynb`) - Market deep dive
3. **Risk Limits** (`01_market_risk_limits.ipynb`) - Position limit framework
4. **Stress Testing** (`02_stress_testing.ipynb`) - Extreme scenario analysis
5. **Manipulation Analysis** (`03_manipulation_sims.ipynb`) - Attack/defense modeling
6. **Interactive Tool** (`interactive_analysis.ipynb`) - Custom analysis

## Project Structure

```
risk-model/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 00_data_exploration.ipynb
│   ├── 01_market_risk_limits.ipynb
│   ├── 02_stress_testing.ipynb
│   ├── 03_manipulation_sims.ipynb
│   ├── risk_summary.ipynb
│   └── interactive_analysis.ipynb
├── src/risk_model/         # Core Python modules
│   ├── binance_data.py     # Binance API integration
│   ├── uniswap_data.py     # Uniswap subgraph queries
│   ├── pricing.py          # Liquidity and volatility analysis
│   ├── metrics.py          # Risk scoring and limits
│   ├── scenarios.py        # Stress testing & manipulation
│   ├── plotting.py         # Professional visualizations
│   ├── volume_analysis.py  # Volume pattern detection
│   ├── notebook_helpers.py # Common notebook functions
│   ├── chart_config.py     # Chart styling configuration
│   ├── scenario_config.py  # Scenario parameters
│   └── interactive_widgets.py # Jupyter widgets
└── config/
    └── markets.yaml        # Market configurations
```

## Key Features

### 1. Real-Time Data Collection
- **Binance API**: Orderbook depth, OHLC history, volume, spreads
- **Market Coverage**: ETH, BTC, SOL, ARB, MATIC, AVAX, etc.
- **Automated Fetching**: Configurable lookback periods and refresh rates

### 2. Advanced Risk Analytics
- **Multi-Factor Scoring**: Liquidity (40%), Volatility (40%), Oracle (20%)
- **Dynamic Limits**: Position limits adjust to market conditions
- **Tiered Access**: Different limits for retail/professional/institutional
- **Real-Time Monitoring**: Utilization tracking with alert levels

### 3. Comprehensive Stress Testing
- **Liquidation Cascades**: Sequential forced closure modeling
- **Gap Risk Analysis**: Sudden price movement impacts
- **Funding Stress**: Extreme rate scenario testing
- **Composite Scenarios**: Market crash, death spiral, black swan events

### 4. Manipulation Analysis
- **Cost Modeling**: Calculate cost to move prices X%
- **Profit Analysis**: Leveraged position profitability matrices
- **Oracle Defense**: Multi-venue aggregation effectiveness
- **Defense Parameters**: Recommended limits and safeguards

### 5. Professional Visualizations
- **Risk Heatmaps**: Color-coded market risk overview
- **Liquidity Curves**: Order book depth visualization
- **Volatility Surfaces**: Multi-timeframe volatility analysis
- **P&L Distributions**: VaR and expected shortfall metrics

## Available Commands

```bash
make help          # Show all available commands
make setup         # Install dependencies and configure environment
make notebook      # Launch Jupyter Lab interface
make run-notebooks # Execute all notebooks with current data
make reports       # Generate HTML reports from notebooks
make verify        # Verify installation and dependencies
make clean         # Clean generated files and cache
```

## Configuration

### Markets Configuration
Edit `risk-model/config/markets.yaml` to:
- Add new perpetual markets
- Adjust oracle configurations
- Set custom risk parameters

### Risk Parameters
Modify `src/risk_model/scenario_config.py` for:
- Stress test scenarios (liquidation %, gap sizes)
- Participant tier limits
- Alert thresholds
- Defense parameters

### Visualization
Customize `src/risk_model/chart_config.py` for:
- Color schemes and styling
- Figure sizes
- Chart templates

## Example Results

When running the framework, you'll see:
- **Risk Scores**: 1-5 scale for each market (1=lowest risk)
- **Position Limits**: Dynamic limits in $MM based on liquidity/volatility
- **Stress Results**: Impact of 10%, 20%, 40% liquidation scenarios
- **Manipulation Costs**: Economic analysis of attack profitability
- **Defense Recommendations**: Specific parameter settings

## Documentation

- **Notebook Guide**: See `NOTEBOOKS_GUIDE.md` for detailed usage
- **Quick Start**: See `QUICKSTART.md` for rapid deployment
- **API Docs**: Inline documentation in each module

## Production Deployment

1. **Automated Execution**: Use `make run-notebooks` in cron/scheduler
2. **Monitoring**: Parse notebook outputs for alerts
3. **Reporting**: Generate daily HTML reports with `make reports`
4. **API Integration**: Import modules directly for programmatic access

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Add tests for new functionality
4. Update relevant notebooks
5. Submit pull request

## License

[Insert your license here]