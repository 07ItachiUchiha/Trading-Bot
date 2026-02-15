# Architecture

High-level overview of how things fit together.

## Layout

```
dashboard/     - Streamlit UI, charts, trade controls
strategy/      - Trading logic (auto trader, news strategy, earnings, etc.)
utils/         - Shared stuff: websockets, sentiment, risk calc, alerts
security/      - Encrypted config storage
data/          - Local JSON/SQLite storage
logs/          - Rotating log files
```

## Data flow

```
Market Data --> WebSocket --> Signal Processing --> Risk Check --> Order Execution --> Dashboard
                               |
News/Events --> Sentiment -----+
```

Basically: price data and news come in, get turned into signals, risk manager decides sizing, and orders go out through Alpaca. The dashboard reads from the same data layer for visualization.

## Key patterns

- **Singleton-ish**: WebSocket connections, config manager, logging setup. Not strict singletons but only instantiated once.
- **Strategy pattern**: Different trading algos (RSI+EMA, Bollinger, news-based) all implement the same interface so they're interchangeable.
- **Observer-ish**: WebSocket callbacks push data to subscribers. Dashboard refreshes on new data.

## Module breakdown

### Strategy layer

- `auto_trading_manager.py` — main orchestrator, runs the loop
- `news_strategy.py` — sentiment-driven entries
- `earnings_report_strategy.py` — trades around earnings events
- `multiple_strategies.py` — manages switching between strategies
- `strategy.py` — base technical strategy (RSI, Bollinger, EMA)

### Data / utils layer

- `market_data_manager.py` — fetches and caches OHLCV from Alpaca
- `websocket_manager.py` — real-time price streaming
- `sentiment_analyzer.py` — VADER-based news scoring
- `signal_processor.py` — combines technical + sentiment signals
- `risk_manager.py` / `risk_management.py` — position sizing, stop losses

### Dashboard

- Streamlit app with pages for manual trading, PnL analysis, trade history, wallet
- Components for charting, position monitoring, correlation checks

## Tech stack

- Python 3.8+
- Pandas / NumPy for number crunching
- Streamlit for the web UI
- Alpaca API for brokerage
- Binance WebSocket as backup data source
- NLTK/VADER for sentiment
- SQLite for local persistence
