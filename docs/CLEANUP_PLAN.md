# Safe Cleanup Plan - Candidate Matrix

## Cleanup Candidate Matrix

| #   | File                               | Risk   | Reason for Removal                                                                      | Imported By                                      | Side Effects         |
| --- | ---------------------------------- | ------ | --------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------- |
| 1   | `api_config.py`                    | LOW    | Dead code, never imported, hardcoded placeholder keys (security risk)                   | Nothing                                          | None                 |
| 2   | `dashboard/pages/page_template.py` | LOW    | Empty template, `display_page()` never called anywhere                                  | Nothing                                          | None                 |
| 3   | `run_patched_trader.py`            | LOW    | Only patches np.NaN for legacy `main.py` which is being removed                         | Nothing                                          | None                 |
| 4   | `main.py`                          | MEDIUM | Legacy Binance bot, uses different broker, `self.llm_model` never defined               | `run_patched_trader.py` (being removed)          | None after batch A   |
| 5   | `utils/binance_websocket.py`       | MEDIUM | Only used by `main.py` Binance pathway                                                  | `main.py` (being removed)                        | None after #4        |
| 6   | `strategy/auto_trader.py`          | MEDIUM | Broken import (`RiskManager` from wrong module), duplicate of `auto_trading_manager.py` | Nothing (crashes on import)                      | None                 |
| 7   | `utils/signal_combiner.py`         | MEDIUM | Duplicate of `signal_processor.py`, has hardcoded demo data                             | `dashboard/components/trading.py` (needs update) | Update 1 import      |
| 8   | `utils/enhanced_sentiment.py`      | HIGH   | Duplicate of `sentiment_analyzer.py`, has extra features to merge first                 | Multiple dashboard components                    | Merge features first |
| 9   | `utils/risk_manager.py`            | HIGH   | Class-based duplicate of `risk_management.py`                                           | `auto_trader.py` (dead)                          | Verify no other refs |

## Batch Removal Plan

### Batch A: Zero-Risk (no importers, no side effects)

**Files:** `api_config.py`, `dashboard/pages/page_template.py`, `run_patched_trader.py`
**Action:** Direct delete
**Verification:** `grep -r "api_config\|page_template\|run_patched" --include="*.py"` returns nothing

### Batch B: Low-to-Medium Risk (importers also being removed)

**Files:** `main.py`, `utils/binance_websocket.py`, `strategy/auto_trader.py`
**Action:** Move to `_quarantine/` folder, delete after 1 week if no issues
**Verification:** All tests pass, `run_auto_trader.py` and `run_bot.py` still work

### Batch C: Requires Migration First (duplicate consolidation)

**Files:** `utils/signal_combiner.py`, `utils/enhanced_sentiment.py`, `utils/risk_manager.py`
**Action:** Merge unique logic into kept module, update imports, then quarantine
**Pre-requisite:** Complete duplicate consolidation (see Architecture doc section 4)

## 3-Stage Process

### Stage 1: Dry Run (this document)

- All candidates identified above
- Import analysis completed
- Risk levels assigned

### Stage 2: Quarantine

- Batch A files deleted directly (zero risk)
- Batch B files moved to `_quarantine/` directory
- Batch C files quarantined after migration

### Stage 3: Permanent Delete

- After 1 week with no issues, delete `_quarantine/` directory
- Verify all tests pass
- Verify dashboard loads correctly
- Verify auto trader runs without errors

## Verification Checklist

- [ ] `python run_bot.py` starts without import errors
- [ ] `python run_auto_trader.py` starts without import errors
- [ ] `python -c "from strategy.auto_trading_manager import AutoTradingManager"` works
- [ ] `python -c "from utils.signal_processor import SignalProcessor"` works
- [ ] `python -c "from utils.sentiment_analyzer import SentimentAnalyzer"` works
- [ ] `python -c "from utils.risk_management import calculate_position_size"` works
- [ ] Dashboard pages load without crashes
- [ ] No `ModuleNotFoundError` in logs after 24h runtime
