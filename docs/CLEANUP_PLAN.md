# Cleanup Plan (Finalized)

Date: 2026-02-15

This document tracks the cleanup effort that transformed the project from a legacy crypto-bot layout into a prediction-first platform.

## Objective

- Remove dead/legacy execution paths
- Keep prediction-centric, testable modules
- Eliminate insecure artifacts
- Maintain working runtime entrypoints (`run_auto_trader.py`, `run_bot.py`)

## Final Actions Completed

### Removed

- `api_config.py`
- `dashboard/pages/page_template.py`
- `run_patched_trader.py`
- `llmApi.txt` (contained exposed keys; replaced with safe docs)

### Legacy Files Removed From Active Runtime

- `main.py` (legacy Binance-centric bot)
- `strategy/auto_trader.py` (duplicate/broken manager)
- `utils/binance_websocket.py` (legacy dependency)
- `utils/signal_combiner.py` (duplicate of signal fusion path)
- `utils/risk_manager.py` (duplicate class-based risk path)

### Replacements / Refactors

- Added prediction-first module:
  - `prediction/engine.py`
  - `prediction/schema.py`
- Updated dashboard market signal panel to use `PredictionEngine`:
  - `dashboard/components/market_data.py`
- Added safe LLM setup guide:
  - `llmApi.md`

## Keep List (Core Runtime)

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

## Verification

- Import checks for key modules pass.
- Prediction tests pass.
- Existing suite passes with expected skips for optional dependencies.

See `docs/CLEANUP_REPORT.md` for execution results and test output summary.
