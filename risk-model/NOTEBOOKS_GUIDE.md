# Risk Model Notebooks Guide

## Overview

This guide explains how to run and use the risk modeling notebooks. The notebooks are designed to be run in a specific order to analyze perpetual market risks comprehensively.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   cd risk-model
   poetry install
   ```

2. **Activate Environment**:
   ```bash
   poetry shell
   ```

3. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

## Notebook Execution Order

### 1. Data Exploration (00_data_exploration.ipynb)
**Purpose**: Initial market analysis and data quality assessment
**Runtime**: ~2-3 minutes

This notebook:
- Fetches current market data from Binance
- Analyzes liquidity depth and volatility
- Creates initial visualizations
- Validates data quality

**Key Outputs**:
- Market summary statistics
- Liquidity curve visualizations
- Volume profile analysis
- Initial risk assessments

### 2. Market Risk Limits (01_market_risk_limits.ipynb)
**Purpose**: Calculate and monitor position limits
**Runtime**: ~1-2 minutes

This notebook:
- Calculates safe position limits
- Implements tiered limits for different participants
- Monitors current utilization
- Provides dynamic adjustments

**Key Outputs**:
- Position limit recommendations
- Utilization monitoring dashboard
- Alert levels for high-risk markets

### 3. Stress Testing (02_stress_testing.ipynb)
**Purpose**: Simulate extreme market conditions
**Runtime**: ~3-4 minutes

This notebook:
- Runs liquidation cascade simulations
- Tests gap risk scenarios
- Analyzes funding stress
- Evaluates composite scenarios

**Key Outputs**:
- Stress test results by market
- Market resilience grades
- Safeguard recommendations

### 4. Manipulation Simulations (03_manipulation_sims.ipynb)
**Purpose**: Analyze manipulation vulnerabilities and defenses
**Runtime**: ~2-3 minutes

This notebook:
- Calculates manipulation costs
- Analyzes profit potential
- Tests oracle defenses
- Recommends protective measures

**Key Outputs**:
- Manipulation cost curves
- Defense effectiveness analysis
- Parameter recommendations

### 5. Executive Summary (risk_summary.ipynb)
**Purpose**: High-level dashboard for decision makers
**Runtime**: ~1 minute

This notebook:
- Aggregates key metrics
- Provides executive overview
- Highlights critical risks
- Summarizes recommendations

**Key Outputs**:
- Executive dashboard
- Risk distribution summary
- Action items

### 6. Interactive Analysis (interactive_analysis.ipynb)
**Purpose**: Deep-dive analysis with interactive widgets
**Runtime**: Interactive

This notebook:
- Provides interactive parameter adjustment
- Enables real-time scenario testing
- Allows custom analysis
- Exports detailed reports

**Key Features**:
- Market selection dropdown
- Parameter sliders
- Real-time chart updates
- Export functionality

## Running All Notebooks

To run all notebooks in sequence:

```bash
# From the notebooks directory
cd risk-model/notebooks

# Run all notebooks
jupyter nbconvert --to notebook --execute 00_data_exploration.ipynb --output 00_data_exploration_executed.ipynb
jupyter nbconvert --to notebook --execute 01_market_risk_limits.ipynb --output 01_market_risk_limits_executed.ipynb
jupyter nbconvert --to notebook --execute 02_stress_testing.ipynb --output 02_stress_testing_executed.ipynb
jupyter nbconvert --to notebook --execute 03_manipulation_sims.ipynb --output 03_manipulation_sims_executed.ipynb
jupyter nbconvert --to notebook --execute risk_summary.ipynb --output risk_summary_executed.ipynb
```

Or use the Makefile:
```bash
make run-notebooks
```

## Configuration

### Market Selection
Edit `config/markets.yaml` to add/remove markets:
```yaml
markets:
  - name: "NEW-PERP"
    binance_symbol_perp: "NEWUSDT"
    oracle: "chainlink"
```

### Risk Parameters
Modify `src/risk_model/scenario_config.py` for:
- Stress test scenarios
- Participant tiers
- Alert thresholds
- Defense parameters

### Chart Styling
Adjust `src/risk_model/chart_config.py` for:
- Color schemes
- Figure sizes
- Font settings
- Grid styles

## Troubleshooting

### API Rate Limits
If you encounter rate limits:
1. Reduce `limit_markets` parameter in notebooks
2. Add delays between API calls
3. Use cached data when available

### Memory Issues
For large datasets:
1. Process markets in batches
2. Clear outputs between notebooks
3. Reduce lookback periods

### Missing Data
If markets have no data:
1. Verify symbol names in config
2. Check API connectivity
3. Ensure markets are active

## Best Practices

1. **Regular Updates**: Run notebooks daily or after significant market events
2. **Version Control**: Commit executed notebooks with timestamps
3. **Parameter Tuning**: Adjust thresholds based on market conditions
4. **Documentation**: Add markdown cells for custom analysis

## Export Options

### PDF Reports
```bash
jupyter nbconvert --to pdf risk_summary.ipynb
```

### HTML Dashboard
```bash
jupyter nbconvert --to html --template lab risk_summary.ipynb
```

### Data Export
Each notebook can export results to CSV:
```python
risk_metrics.to_csv('risk_metrics_20231201.csv')
```

## Support

For issues or questions:
1. Check notebook markdown cells for detailed explanations
2. Review error messages in output cells
3. Verify all dependencies are installed
4. Ensure API keys/access are configured