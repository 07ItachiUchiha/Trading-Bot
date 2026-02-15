# API Reference

Quick reference for the main classes and functions. Not exhaustive — check the source for edge cases.

---

## AutoTradingManager

Main loop that runs the strategies, checks signals, and submits orders.

```python
class AutoTradingManager:
    def __init__(self, symbols, timeframe, capital=10000, 
                 risk_percent=1.0, profit_target_percent=3.0,
                 daily_profit_target=5.0, use_news=True, 
                 news_weight=0.5, use_earnings=True, 
                 earnings_weight=0.6)
```

**Methods:**

- `run()` — kicks off the trading loop. Blocks until stopped.
- `stop()` — graceful shutdown.
- `get_historical_data(symbol, timeframe, days_back=30)` — pulls OHLCV bars from Alpaca. Returns a DataFrame.

---

## SentimentAnalyzer

Runs news through VADER + some custom weighting to produce a buy/sell/neutral signal.

```python
class SentimentAnalyzer:
    def __init__(self, api_keys=None, config=None)
```

**Methods:**

- `analyze_sentiment(symbol, news_items=None, days_back=3)` — returns a dict with `signal`, `confidence` (0-1), `score` (-1 to 1), `reasoning`, `news_count`.
- `fetch_news(symbol, days_back=3)` — grabs articles from NewsAPI / Finnhub.

---

## RiskManager

Position sizing (Kelly criterion + volatility scaling) and trade validation.

```python
class RiskManager:
    def __init__(self, config=None)
```

**Methods:**

- `calculate_position_size(symbol, entry_price, stop_loss, confidence=0.7, account_value=None)` — returns dict with `size`, `risk_amount`, `max_loss_percent`.
- `validate_trade(symbol, size, direction, current_positions)` — returns True/False based on risk limits.

---

## SignalProcessor

Combines technical, sentiment, and earnings signals with configurable weights.

```python
class SignalProcessor:
    def __init__(self, config=None)
```

**Methods:**

- `process_signals(symbol, technical_signal=None, sentiment_signal=None, earnings_signal=None, price_data=None)` — merges signals, returns dict with `signal`, `confidence`, `reasoning`, `timestamp`.

---

## WebSocketManager

Wraps Alpaca/Binance websocket connections for real-time price data.

```python
class WebSocketManager:
    def __init__(self, api_key, api_secret, base_url)
```

**Methods:**

- `subscribe(symbol, callback)` — start streaming data for a symbol.
- `unsubscribe(symbol, callback=None)` — stop streaming.

---

## Utility functions

### `standardize_symbol(symbol, destination='alpaca')`
Converts between exchange formats (Alpaca `BTC/USD`, Binance `BTCUSDT`, etc).

### `calculate_position_size(capital, risk_percent, entry_price, stop_loss, max_capital_per_trade=0.15)`
Simple risk-based sizing. Caps at `max_capital_per_trade` fraction of the account.

### `calculate_trailing_stop(current_price, direction, atr, profit_ticks, initial_stop)`
ATR-based trailing stop that tightens as the trade moves in your favor.

---

## Config objects

Typical trading config:
```python
{
    'symbols': ['BTC/USD', 'ETH/USD'],
    'timeframe': '1H',
    'capital': 10000,
    'risk_percent': 1.0,
    'profit_target_percent': 3.0,
    'daily_profit_target': 5.0,
    'max_capital_per_trade': 0.15
}
```

Strategy weights:
```python
{
    'use_news': True,
    'news_weight': 0.5,
    'use_earnings': True,
    'earnings_weight': 0.6,
    'sentiment_threshold': 0.7,
    'technical_weight': 0.4
}
```

## Common errors

- **APIError** — rate limits or bad credentials
- **DataError** — missing/corrupt market data
- **RiskError** — trade exceeds a risk limit
- **ConnectionError** — network issues, websocket drops
