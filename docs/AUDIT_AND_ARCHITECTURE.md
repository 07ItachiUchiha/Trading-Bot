# Audit and Architecture (Current State)

Date: 2026-02-15

## Platform Positioning

The codebase is now structured as an **Algorithmic Market Prediction Platform**:

- Prediction-first output (`BUY` / `SELL` / `HOLD`)
- Confidence scoring and factor attribution
- NLP/sentiment + technical + event fusion
- Explainable rationale generation with deterministic fallback
- Fast path support with latency capture

## Core Runtime Architecture

```
Market Data (REST/WebSocket) + News/Event Feeds
              |
              v
      Feature + Signal Extraction
   (technical, sentiment, earnings)
              |
              v
      prediction/engine.py
              |
              v
 prediction/schema.py (contracted output)
              |
              +--> strategy/prediction_runtime_manager.py (orchestration)
              +--> dashboard/components/market_data.py (display)
              +--> alerts/logging
```

## Active Module Classification

### Core

- `prediction/engine.py`
- `prediction/schema.py`
- `strategy/prediction_runtime_manager.py`
- `utils/signal_processor.py`
- `utils/sentiment_analyzer.py`
- `utils/risk_management.py`
- `utils/market_data_manager.py`
- `dashboard/app.py`
- `run_auto_trader.py`
- `run_bot.py`

### Supporting

- `strategy/news_strategy.py`
- `strategy/earnings_report_strategy.py`
- `utils/news_fetcher.py`
- `utils/llm_integration.py`
- `utils/finnhub_webhook.py`
- `security/secure_config.py`
- dashboard pages/components that render UI

### Removed / Deprecated

These paths were removed from active runtime because they were duplicate, legacy, or inconsistent with prediction-first architecture:

- `main.py`
- `strategy/auto_trader.py`
- `utils/binance_websocket.py`
- `utils/signal_combiner.py`
- `utils/risk_manager.py`
- `run_patched_trader.py`
- `api_config.py`
- `dashboard/pages/page_template.py`

## Prediction Contract

Predictions are emitted via `Prediction` dataclass:

- `symbol`
- `timestamp`
- `action`
- `confidence`
- `horizon`
- `factor_breakdown`
- `rationale`
- `risk_flags`
- `latency_ms`

This output contract is exercised in `tests/test_prediction.py`.

## Operational Notes

- LLM usage is optional (`GEMINI_API_KEY` or `OPENROUTER_API_KEY`).
- Missing LLM keys do not block predictions; deterministic rationale fallback is used.
- Sensitive key examples are documented safely in `llmApi.md` (no hardcoded secrets).

## Verification Snapshot

- `python -m pytest tests/test_prediction.py -q` -> pass
- `python -m pytest tests/test_suite.py -q` -> pass (with expected skips)
