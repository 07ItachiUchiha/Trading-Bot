# Architecture

High-level view of the current prediction-first platform.

## Layout

```
prediction/    - Prediction schema + fusion engine
strategy/      - Orchestration + strategy signal producers
utils/         - Data, sentiment, risk, LLM, webhook utilities
dashboard/     - Streamlit UI and prediction visualization
security/      - Secure config helpers
tests/         - Prediction + regression tests
```

## Data flow

```
Market Data + News/Events --> Signal Extraction --> Prediction Engine --> Dashboard/Alerts
                                         |
                                         +--> factor breakdown + rationale + risk flags
```

Predictions are generated from technical + sentiment + event factors, then exposed with confidence and explanation.

## Key patterns

- **Prediction contract**: structured `Prediction` output across runtime paths.
- **Modular fusion**: independent factor analyzers contribute weighted signals.
- **Fail-safe behavior**: missing external services fall back to deterministic outputs.
- **Observer-style updates**: dashboard refreshes from data and prediction updates.

## Module breakdown

### Core prediction/orchestration

- `prediction/engine.py` — multi-factor prediction fusion
- `prediction/schema.py` — prediction output dataclass contract
- `strategy/auto_trading_manager.py` — runtime loop and signal orchestration

### Data and analytics utilities

- `utils/market_data_manager.py` — OHLCV fetching/caching
- `utils/websocket_manager.py` — real-time updates
- `utils/sentiment_analyzer.py` — NLP sentiment scoring
- `utils/signal_processor.py` — weighted signal combination helpers
- `utils/risk_management.py` — sizing/trailing-stop helpers
- `utils/llm_integration.py` — optional LLM rationale provider

### Dashboard

- `dashboard/app.py` — Streamlit entrypoint
- `dashboard/components/market_data.py` — prediction-based market signal view
- other components/pages for controls, PnL, history, and monitoring

## Tech stack

- Python 3.8+
- Pandas / NumPy
- Streamlit
- Alpaca API
- NLTK/VADER + TextBlob for sentiment features
- Optional LLM provider: Gemini or OpenRouter
- SQLite for local persistence
