# Docs

Project documentation. These are mostly for my own reference.

- [Architecture](ARCHITECTURE.md) — how the modules fit together
- [API Reference](API_REFERENCE.md) — main classes and functions
- [Deployment](DEPLOYMENT.md) — setup guide for local and production

## Quick start

```bash
pip install -r requirements.txt
cp config_example.py config.py
# fill in your API keys
python setup_check.py
streamlit run dashboard/app.py
```

## Trading strategies

The bot supports a few different approaches:

**RSI + EMA crossover** — classic momentum. RSI flags overbought/oversold, EMA cross confirms direction, volume validates. Works ok on 1H timeframes.

**Bollinger Bands + RSI** — mean reversion. Looks for price touching the bands with RSI divergence. Better in ranging markets.

**News sentiment** — pulls articles from NewsAPI and Finnhub, scores them with VADER, generates signals when sentiment is strong enough. Weighted by source count and recency.

**Earnings events** — watches for upcoming earnings reports, positions based on expected surprise and recent momentum. Monitors 3 days before through 2 days after.

## Risk management

- Position sizing uses a modified Kelly criterion with volatility adjustment
- Stop losses are ATR-based and trail as the trade moves
- Daily loss limit: 5% (configurable)
- Correlation protection: won't pile into correlated assets
- Max 15% of account on any single trade

## Troubleshooting

1. Run `python setup_check.py` first
2. Check `logs/` for detailed errors
3. Make sure API keys are set (Alpaca is required, others optional)
4. Run the test suite: `python tests/test_suite.py`
