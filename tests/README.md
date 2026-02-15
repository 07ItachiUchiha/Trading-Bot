# Tests

Basic test suite for the trading bot. Covers the main modules but doesn't aim for 100% coverage.

## Running

```bash
python tests/test_suite.py
```

Or to run a specific test class:
```bash
python -m unittest tests.test_suite.TestTradingStrategies
```

## What's tested

- **Strategy calculations** — Bollinger Bands, RSI, indicator math
- **Risk management** — position sizing, stop loss levels
- **Data validation** — OHLCV format checks, market data fetching (mocked)
- **WebSocket** — data standardization
- **Sentiment** — scoring positive/negative news
- **Database** — that the trade storage functions exist and are callable
- **Config** — required attributes exist and have sane values
- **News API** — integration with mocked HTTP responses

Most external API calls are mocked, so you can run the tests offline.

## Adding tests

Just add a new `TestCase` class in `test_suite.py` and add it to the `test_classes` list
in `run_comprehensive_tests()`. Keep test methods short — one assertion per concept.

## Coverage (optional)

```bash
pip install coverage
coverage run tests/test_suite.py
coverage report
```
