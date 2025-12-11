# Quick Start Guide

## 🚀 Getting Started in 3 Minutes

```bash
# 1. Clone and setup
git clone <repository>
cd clob-risk-modelling
make setup

# 2. Run all analysis notebooks
make run-notebooks

# 3. View results in risk-model/notebooks/executed/
```

## 📊 Interactive Analysis

For hands-on exploration:
```bash
make notebook
```

Then open:
1. **`risk_summary.ipynb`** - Executive dashboard
2. **`interactive_analysis.ipynb`** - Real-time parameter adjustment
3. **`00_data_exploration.ipynb`** - Deep market analysis

## 📈 What You'll See

### Executive Summary Dashboard
- Market risk scores (1-5 scale) with color coding
- Current position limits and utilization
- Key metrics: liquidity, volatility, max OI
- Action items and recommendations

### Risk Analysis Features
1. **Position Limits**
   - Dynamic limits based on market conditions
   - Tiered access (Retail/Professional/Institutional)
   - Real-time utilization monitoring
   - Alert system for high-risk situations

2. **Stress Testing**
   - Liquidation cascade simulations (10%, 20%, 40% OI)
   - Gap risk analysis (5%, 10%, 20% instant moves)
   - Funding rate stress scenarios
   - Composite "Black Swan" events

3. **Manipulation Analysis**
   - Cost to manipulate prices by X%
   - Profitability analysis with leverage
   - Oracle defense effectiveness
   - Recommended safeguards

### Interactive Tools
- Market selection dropdown
- Parameter sliders (leverage, position size, etc.)
- Real-time chart updates
- Scenario comparison tools
- Export functionality

## 🔧 Configuration

### Add New Markets
Edit `risk-model/config/markets.yaml`:
```yaml
- name: "NEW-PERP"
  binance_symbol_perp: "NEWUSDT"
  oracle: "chainlink"
```

### Adjust Risk Parameters
Edit `risk-model/src/risk_model/scenario_config.py`:
- Stress scenarios
- Alert thresholds  
- Defense parameters
- Participant tiers

## 📊 Example Output

Running the framework produces:

### Risk Metrics
```
Market    Risk Score    Max OI ($MM)    Position Limit ($MM)    Current Util %
ETH-PERP     2.1          125.5            25.1                   35.2%
BTC-PERP     1.8          450.2            90.0                   42.1%
SOL-PERP     3.2           45.3             9.1                   68.5%
```

### Stress Test Results
```
Scenario: 20% OI Liquidation
- ETH: 8.5% price impact, 85 bps recovery cost
- BTC: 5.2% price impact, 52 bps recovery cost  
- SOL: 15.3% price impact, 153 bps recovery cost
```

### Manipulation Analysis
```
5% Price Manipulation Cost:
- ETH: $2.5MM (profitable with 25x leverage, $5MM position)
- BTC: $8.1MM (requires 50x leverage for profitability)
- SOL: $0.8MM (highly vulnerable to manipulation)
```

## ⚡ Performance Tips

1. **Faster Execution**: Limit markets for quicker analysis
   ```python
   market_data = fetch_market_data(markets, limit_markets=5)
   ```

2. **Reduce API Calls**: Use shorter lookback periods
   ```python
   lookback_days = 7  # Instead of 30
   ```

3. **Memory Optimization**: Process markets in batches
   ```python
   for market_batch in chunks(markets, 10):
       process_batch(market_batch)
   ```

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Import errors | Run `poetry shell` first |
| API rate limits | Increase `rate_limit_delay` in binance_data.py |
| Memory issues | Reduce `lookback_days` or `limit_markets` |
| Missing data | Check market symbols in config/markets.yaml |
| Notebook kernel dies | Reduce data size or clear outputs |

### Quick Fixes

```bash
# Reset environment
make clean
make setup

# Fix Poetry issues  
make poetry-fix
make setup

# Verify installation
make verify
```

## 📚 Learn More

- **Full Documentation**: See README.md
- **Notebook Details**: See NOTEBOOKS_GUIDE.md
- **API Reference**: Check module docstrings
- **Configuration**: Review scenario_config.py

## 💡 Pro Tips

1. **Daily Reports**: Schedule `make run-notebooks` via cron
2. **Custom Alerts**: Parse executed notebooks for limit breaches
3. **API Integration**: Import risk_model modules directly
4. **Version Control**: Commit executed notebooks with timestamps

Ready to analyze perpetual market risks! 🚀