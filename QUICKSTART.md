# Quick Start Guide

## Getting Started

1. **Install dependencies:**
   ```bash
   make setup
   ```

2. **Launch Jupyter notebooks:**
   ```bash
   make notebook
   ```

3. **Open the notebooks:**
   - `risk_summary.ipynb` - Executive dashboard with professional charts
   - `interactive_analysis.ipynb` - Interactive tool for custom analysis
   - `00_data_exploration.ipynb` - Detailed market data exploration

## Key Features

### Professional Charts
- **Risk Heatmaps** - Color-coded risk scores across markets
- **Volatility Surfaces** - Multi-asset volatility dynamics with confidence bands
- **P&L Distributions** - PDF/CDF analysis with VaR metrics
- **Volume-Depth Correlations** - Market quality assessments
- **Liquidity Depth Curves** - Order book visualization

### Interactive Analysis
- Real-time parameter adjustment
- Custom stress testing scenarios
- Export functionality for reports
- Volume pattern analysis

## What's Working

**Binance API Integration**
- Order book snapshots
- Historical OHLC data
- 24h volume statistics
- Real-time spreads

**Risk Analytics**
- Liquidity scoring
- Volatility analysis
- Position limit calculations
- Stress test scenarios

**Professional Visualizations**
- Clean, gradient-based color schemes
- Publication-ready charts
- Interactive widgets

## Known Limitations

- Uniswap API requires updated endpoint (The Graph deprecated old URL)
- AMM liquidity analysis is simplified
- Historical depth data is simulated

## Example Output

When you run the summary notebook, you'll see:
- Current prices and volumes for ETH, BTC, SOL
- Risk scores (1-5 scale) for each market
- Recommended OI and position limits
- Stress test results
- Professional charts with clean styling

## Customization

Edit `risk-model/config/markets.yaml` to:
- Add new markets
- Adjust risk parameters
- Change API endpoints

## Next Steps

1. Run the notebooks to see live data
2. Adjust risk parameters in the interactive tool
3. Export reports for operational use
4. Set up automated monitoring

## Troubleshooting

### Common Issues

**Import errors**: Make sure you're in the poetry environment
```bash
poetry shell
```

**API rate limits**: The code includes delays but you may need to adjust
```python
self.rate_limit_delay = 0.05  # Increase if needed
```

**Memory issues**: Reduce lookback period or limit number of markets
```python
lookback_days = 7  # Instead of 30
```

### Support

For issues or questions:
1. Check the error messages in Jupyter output
2. Review the API documentation for Binance
3. Ensure all dependencies are installed with `poetry install`