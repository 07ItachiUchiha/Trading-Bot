# API Reference

Reference for the currently active prediction-first modules.

## Prediction

### `Prediction` (`prediction/schema.py`)

Structured output contract for every market prediction.

```python
Prediction(
    symbol: str,
    timestamp: str = "",
    action: str = "HOLD",
    confidence: float = 0.0,
    horizon: str = "1h",
    factor_breakdown: dict = {},
    rationale: str = "",
    risk_flags: list = [],
    latency_ms: float = 0.0,
    metadata: dict = {}
)
```

Key methods:

- `to_dict() -> dict`
- `to_json(indent=2) -> str`
- `from_dict(data: dict) -> Prediction`
- `summary() -> str`

Notes:

- `action` is normalized to `BUY` / `SELL` / `HOLD`.
- `confidence` is clamped to `[0.0, 1.0]`.

### `PredictionEngine` (`prediction/engine.py`)

Fuses technical, sentiment, earnings, and optional LLM rationale into one `Prediction`.

```python
class PredictionEngine:
    def __init__(self, config: Optional[dict] = None)
    def predict(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sentiment_override: Optional[dict] = None,
        earnings_override: Optional[dict] = None,
        horizon: str = "1h",
    ) -> Prediction
```

Behavior:

- Produces action + confidence + factor breakdown + rationale + risk flags + latency.
- Uses deterministic fallback rationale if LLM integration is unavailable.

## Runtime Orchestration

### `AutoTradingManager` (`strategy/auto_trading_manager.py`)

Main runtime loop that updates data, computes signals, and emits predictions.

```python
class AutoTradingManager:
    def __init__(
        self,
        symbols,
        timeframe,
        capital=10000,
        risk_percent=1.0,
        profit_target_percent=3.0,
        daily_profit_target=5.0,
        use_news=True,
        news_weight=0.5,
        use_earnings=True,
        earnings_weight=0.6,
        signal_only=True,
    )
```

Key methods:

- `run()` - starts blocking trading/prediction loop.
- `run_cycle()` - runs one deterministic cycle over all symbols.
- `stop()` - graceful shutdown.
- `process_news_event(payload)` - webhook callback hook.
- `process_earnings_event(payload)` - webhook callback hook.
- `get_historical_data(symbol, timeframe, days_back=30)` - returns OHLCV dataframe.
- `get_last_signal()` - latest combined signal structures by symbol.
- `get_latest_predictions()` - latest `Prediction` objects serialized to dict.

## Signal and Risk Utilities

### `SignalProcessor` (`utils/signal_processor.py`)

Weighted signal combiner for technical/sentiment/earnings dictionaries.

```python
class SignalProcessor:
    def __init__(self, config=None)
    def process_signals(
        self,
        symbol,
        technical_signal=None,
        sentiment_signal=None,
        earnings_signal=None,
        price_data=None,
    ) -> dict
    def get_all_active_signals() -> dict
    def cleanup_expired_signals() -> int
```

### Risk Functions (`utils/risk_management.py`)

- `calculate_position_size(capital, risk_percent, entry_price, stop_loss, max_capital_per_trade=0.15) -> float`
- `calculate_trailing_stop(current_price, direction, atr, profit_ticks, initial_stop) -> float`
- `manage_open_position(position, current_price, current_atr) -> tuple[action, new_stop, exit_size]`

## Data and Connectivity

### `WebSocketManager` (`utils/websocket_manager.py`)

Alpaca streaming manager with reconnect/backoff logic.

```python
class WebSocketManager:
    @classmethod
    def get_instance(cls, api_key, api_secret, base_url="wss://stream.data.alpaca.markets/v2")
    def start()
    def stop()
    def subscribe(symbol, callback)
    def unsubscribe(symbol, callback=None)
```

### Symbol Helpers (`utils/symbol_helper.py`)

- `standardize_symbol(symbol, destination='alpaca')`
- `get_base_currency(symbol)`
- `get_quote_currency(symbol)`
- `is_crypto(symbol)`
- `get_initial_price(symbol)`

## LLM Integration

### `get_llm_response` (`utils/llm_integration.py`)

Generates rationale text via configured provider.

Expected environment configuration:

- `GEMINI_API_KEY`, or
- `OPENROUTER_API_KEY`

If neither key is set, callers should handle fallback behavior (already done in `PredictionEngine`).
