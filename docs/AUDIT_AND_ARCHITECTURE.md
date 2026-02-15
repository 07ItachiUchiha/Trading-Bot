# Algorithmic Market Prediction Platform - Audit and Architecture

## 1. Current-State Audit Table

| File                                             | Classification | Lines | Purpose                                               | Issues                                                   |
| ------------------------------------------------ | -------------- | ----- | ----------------------------------------------------- | -------------------------------------------------------- |
| `config.py`                                      | CORE           | 100   | Central config, env vars, trading defaults            | EMA_SHORT=50 drifts from config_manager (12)             |
| `config_manager.py`                              | CORE           | 266   | Secure config loading with fallback chain             | Config drift with config.py                              |
| `config_example.py`                              | SUPPORTING     | ~20   | Template for new users                                | None                                                     |
| `main.py`                                        | LEGACY         | 434   | Binance-based VolatilityBreakoutBot                   | `self.llm_model` never defined, wrong broker             |
| `run_bot.py`                                     | CORE           | ~30   | Entry point for dashboard                             | None                                                     |
| `run_auto_trader.py`                             | CORE           | ~50   | Entry point for auto trading                          | None                                                     |
| `run_patched_trader.py`                          | LEGACY         | ~10   | np.NaN patch for main.py                              | Only needed by dead main.py                              |
| `patches.py`                                     | SUPPORTING     | ~20   | Numpy compatibility patches                           | Still useful                                             |
| `setup_check.py`                                 | SUPPORTING     | ~80   | Dependency checker                                    | None                                                     |
| `api_config.py`                                  | DEAD           | ~35   | Hardcoded placeholder API keys                        | Never imported, security risk                            |
| **strategy/**                                    |                |       |                                                       |                                                          |
| `strategy/strategy.py`                           | CORE           | 196   | Technical analysis (consolidation/breakout detection) | Uses deprecated fillna(method=)                          |
| `strategy/auto_trading_manager.py`               | CORE           | 720   | Main automated trading loop                           | Production entry point                                   |
| `strategy/auto_trader.py`                        | LEGACY         | 577   | Broken duplicate of auto_trading_manager              | Import `RiskManager` from wrong module                   |
| `strategy/news_strategy.py`                      | CORE           | ~250  | News-based signal generation                          | None                                                     |
| `strategy/earnings_report_strategy.py`           | SUPPORTING     | ~200  | Earnings calendar signals                             | Placeholder API URL                                      |
| `strategy/multiple_strategies.py`                | SUPPORTING     | ~150  | Multi-strategy combiner                               | None                                                     |
| **utils/**                                       |                |       |                                                       |                                                          |
| `utils/signal_processor.py`                      | CORE           | 190   | Weighted signal combination                           | Duplicate of signal_combiner                             |
| `utils/signal_combiner.py`                       | LEGACY         | 190   | Duplicate signal combiner                             | Hardcoded demo data fallback                             |
| `utils/risk_management.py`                       | CORE           | 100   | Position sizing, trailing stops                       | Standalone functions, well-structured                    |
| `utils/risk_manager.py`                          | SUPPORTING     | ~200  | Class-based risk management                           | Duplicate of risk_management                             |
| `utils/sentiment_analyzer.py`                    | CORE           | 440   | Multi-source news sentiment (VADER + TextBlob)        | `url` undefined in `_fetch_from_finnhub()`               |
| `utils/enhanced_sentiment.py`                    | SUPPORTING     | ~300  | Extended sentiment with more sources                  | Hardcoded test API keys, duplicate of sentiment_analyzer |
| `utils/llm_integration.py`                       | CORE           | 30    | OpenRouter LLM wrapper                                | Minimal, needs expansion                                 |
| `utils/websocket_manager.py`                     | CORE           | 666   | Alpaca WebSocket with auto-reconnect                  | Syntax error in `_should_rate_limit()`                   |
| `utils/market_data_manager.py`                   | CORE           | ~350  | Multi-source data fetching + caching                  | `data_source` key missing, `winsdow` typo                |
| `utils/news_fetcher.py`                          | SUPPORTING     | ~200  | Raw news fetching                                     | None                                                     |
| `utils/news_trading_advisor.py`                  | SUPPORTING     | ~150  | News-based trade recommendations                      | None                                                     |
| `utils/connection_limiter.py`                    | SUPPORTING     | 80    | Singleton rate limiter                                | BUG: `limit` undefined                                   |
| `utils/finnhub_webhook.py`                       | SUPPORTING     | 112   | Flask webhook for Finnhub                             | BUG: `hmac.new()` should be `hmac.HMAC()`                |
| `utils/telegram_alert.py`                        | SUPPORTING     | ~80   | Telegram notifications                                | None                                                     |
| `utils/discord_webhook.py`                       | SUPPORTING     | ~60   | Discord notifications                                 | None                                                     |
| `utils/slack_webhook.py`                         | SUPPORTING     | ~60   | Slack notifications                                   | None                                                     |
| `utils/symbol_helper.py`                         | SUPPORTING     | ~40   | Symbol normalization                                  | None                                                     |
| `utils/export.py`                                | SUPPORTING     | ~80   | Trade data export                                     | None                                                     |
| `utils/logging_config.py`                        | SUPPORTING     | ~50   | Logging setup                                         | None                                                     |
| `utils/dataframe_utils.py`                       | SUPPORTING     | ~30   | DataFrame helpers                                     | None                                                     |
| `utils/binance_websocket.py`                     | LEGACY         | 302   | Binance WebSocket client                              | Only used by dead main.py                                |
| **dashboard/**                                   |                |       |                                                       |                                                          |
| `dashboard/app.py`                               | CORE           | ~200  | Streamlit dashboard entry                             | Import path bug                                          |
| `dashboard/xau_handler.py`                       | SUPPORTING     | ~100  | Gold/XAU-specific handling                            | Hardcoded `API_KEY = "demo"`                             |
| `dashboard/components/dashboard_ui.py`           | CORE           | ~300  | Main dashboard layout                                 | `export_to_excel/sheets` wrong arg count                 |
| `dashboard/components/trading.py`                | CORE           | ~250  | Trading interface                                     | `st.rerun()` at import time                              |
| `dashboard/components/market_data.py`            | CORE           | ~200  | Market data display                                   | None                                                     |
| `dashboard/components/auth.py`                   | SUPPORTING     | ~100  | JWT authentication                                    | SECRET_KEY defaults empty                                |
| `dashboard/components/database.py`               | SUPPORTING     | ~150  | SQLite storage                                        | Stores API keys in plaintext                             |
| `dashboard/components/risk_management.py`        | SUPPORTING     | ~100  | Dashboard risk display                                | Duplicate of utils/risk_management                       |
| `dashboard/components/position_monitor.py`       | SUPPORTING     | ~120  | Position tracking display                             | None                                                     |
| `dashboard/components/pnl_visualization.py`      | SUPPORTING     | ~150  | PnL charts                                            | None                                                     |
| `dashboard/components/news_analysis.py`          | SUPPORTING     | ~100  | News display component                                | None                                                     |
| `dashboard/components/strategy_selector.py`      | SUPPORTING     | ~80   | Strategy picker UI                                    | None                                                     |
| `dashboard/components/trade_controls.py`         | SUPPORTING     | ~100  | Trade execution controls                              | None                                                     |
| `dashboard/components/trade_filter.py`           | SUPPORTING     | ~60   | Trade filtering                                       | None                                                     |
| `dashboard/components/wallet.py`                 | SUPPORTING     | ~80   | Wallet display                                        | None                                                     |
| `dashboard/components/position_utils.py`         | SUPPORTING     | ~40   | Position helpers                                      | None                                                     |
| `dashboard/components/correlation_protection.py` | SUPPORTING     | ~80   | Correlation-based risk                                | None                                                     |
| `dashboard/pages/manual_trading.py`              | SUPPORTING     | ~100  | Manual trade page                                     | None                                                     |
| `dashboard/pages/news_analysis.py`               | SUPPORTING     | ~80   | News page                                             | None                                                     |
| `dashboard/pages/pnl_analysis.py`                | SUPPORTING     | ~80   | PnL page                                              | `st.set_page_config()` at top level                      |
| `dashboard/pages/trade_history.py`               | SUPPORTING     | ~80   | Trade history page                                    | `st.set_page_config()` at top level                      |
| `dashboard/pages/wallet.py`                      | SUPPORTING     | ~60   | Wallet page                                           | `st.set_page_config()` at top level                      |
| `dashboard/pages/strategy_tester.py`             | SUPPORTING     | ~100  | Strategy backtester page                              | None                                                     |
| `dashboard/pages/page_template.py`               | DEAD           | 22    | Empty template                                        | `display_page()` never called                            |
| **other/**                                       |                |       |                                                       |                                                          |
| `security/secure_config.py`                      | CORE           | ~200  | Encrypted config storage                              | None                                                     |
| `backtest/backtest_strategy.py`                  | SUPPORTING     | ~150  | Backtesting engine                                    | None                                                     |
| `backtest/run_backtest.py`                       | SUPPORTING     | ~50   | Backtest runner                                       | None                                                     |
| `tests/test_suite.py`                            | SUPPORTING     | ~200  | Test suite                                            | Expects dict but gets tuple                              |

## 2. Target Architecture

```
+---------------------------------------------------------------------------+
|                  ALGORITHMIC MARKET PREDICTION PLATFORM                     |
+---------------------------------------------------------------------------+
|                                                                           |
|  Data Layer                 Prediction Layer         Output Layer          |
|  +-----------------+       +-------------------+    +------------------+  |
|  | Market Data     |       | Prediction Engine |    | REST API         |  |
|  | (Alpaca REST +  |------>| (prediction/      |--->| (future)         |  |
|  |  WebSocket)     |       |  engine.py)       |    +------------------+  |
|  +-----------------+       |                   |    | Streamlit        |  |
|  | News Fetcher    |------>| Combines:         |--->| Dashboard        |  |
|  | (multi-source)  |       |  - Technical      |    +------------------+  |
|  +-----------------+       |  - Sentiment/NLP  |    | Notifications    |  |
|  | Earnings Cal.   |------>|  - Earnings       |--->| (Telegram/       |  |
|  +-----------------+       |  - LLM Reasoning  |    |  Discord/Slack)  |  |
|                            +-------------------+    +------------------+  |
|                                    |                                      |
|                            +-------v-------+                              |
|                            | Prediction    |                              |
|                            | Output Schema |                              |
|                            | (see below)   |                              |
|                            +---------------+                              |
|                                                                           |
|  Support Layer                                                            |
|  +-----------------+  +------------------+  +------------------+          |
|  | Risk Manager    |  | Config Manager   |  | Connection       |          |
|  | (position size, |  | (secure config,  |  | Limiter          |          |
|  |  trailing stop) |  |  env fallback)   |  | (rate limiting)  |          |
|  +-----------------+  +------------------+  +------------------+          |
|                                                                           |
+---------------------------------------------------------------------------+
```

## 3. Prediction Output Schema

```python
@dataclass
class Prediction:
    symbol: str              # e.g. "BTC/USD"
    timestamp: str           # ISO-8601
    action: str              # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0 to 1.0
    horizon: str             # "1h" | "4h" | "1d"
    factor_breakdown: dict   # {"technical": 0.6, "sentiment": 0.3, ...}
    rationale: str           # Human-readable explanation
    risk_flags: list         # ["high_volatility", "low_liquidity", ...]
    latency_ms: float        # Time to produce this prediction
```

## 4. Duplicate Module Consolidation Plan

| Keep                               | Remove                                    | Migration                                          |
| ---------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| `utils/signal_processor.py`        | `utils/signal_combiner.py`                | Merge unique logic into signal_processor           |
| `utils/sentiment_analyzer.py`      | `utils/enhanced_sentiment.py`             | Merge enhanced features into sentiment_analyzer    |
| `utils/risk_management.py`         | `utils/risk_manager.py`                   | Keep standalone functions, wrap in class if needed |
| `strategy/auto_trading_manager.py` | `strategy/auto_trader.py`                 | auto_trader is already broken (bad import)         |
| `utils/risk_management.py`         | `dashboard/components/risk_management.py` | Dashboard version calls into utils version         |

## 5. File-by-File Migration Plan

### Phase 1: Cleanup (remove dead/legacy code)

1. Delete `api_config.py` (dead, security risk)
2. Delete `dashboard/pages/page_template.py` (empty template)
3. Delete `run_patched_trader.py` (only patched dead main.py)
4. Quarantine `main.py` (legacy Binance bot)
5. Quarantine `utils/binance_websocket.py` (only used by main.py)
6. Quarantine `strategy/auto_trader.py` (broken duplicate)
7. Quarantine `utils/signal_combiner.py` (duplicate)

### Phase 2: P0 Bug Fixes

1. Fix `websocket_manager.py` indentation error in `_should_rate_limit()`
2. Fix `connection_limiter.py` undefined `limit` variable
3. Fix `finnhub_webhook.py` `hmac.new()` call
4. Fix `sentiment_analyzer.py` undefined `url` in `_fetch_from_finnhub()`
5. Fix `dashboard/app.py` import path
6. Resolve config drift between config.py and config_manager.py

### Phase 3: New Prediction Engine

1. Create `prediction/` package
2. Build `prediction/engine.py` (core prediction pipeline)
3. Build `prediction/schema.py` (Prediction dataclass)
4. Refactor `signal_processor.py` to feed into prediction engine
5. Expand `llm_integration.py` to support Gemini + explain predictions

### Phase 4: Integration

1. Wire prediction engine into auto_trading_manager
2. Update dashboard to show predictions with explainability
3. Add latency tracking
4. Write acceptance tests

## 6. LLM Recommendation

**Gemini API is suitable** for this use case. Google's Gemini API offers:

- Free tier with generous rate limits (15 RPM for gemini-pro)
- Good at structured output (JSON predictions)
- Strong reasoning for market analysis explanations

**Implementation plan**: Add Gemini as primary LLM, keep OpenRouter as fallback.
The LLM will be used for:

- Generating human-readable rationale for each prediction
- Synthesizing multiple signals into coherent explanations
- Identifying risk flags from market context

## 7. Latency Budgets

| Component           | Target   | Notes                                 |
| ------------------- | -------- | ------------------------------------- |
| Data fetch (cached) | < 50ms   | Local cache hit                       |
| Data fetch (API)    | < 500ms  | Alpaca REST call                      |
| Technical analysis  | < 100ms  | NumPy/Pandas ops on buffered data     |
| Sentiment scoring   | < 200ms  | VADER + TextBlob on cached news       |
| LLM reasoning       | < 3000ms | Gemini API call (async, non-blocking) |
| Total prediction    | < 4000ms | End-to-end with LLM                   |
| Total (no LLM)      | < 1000ms | Fast path without rationale           |
