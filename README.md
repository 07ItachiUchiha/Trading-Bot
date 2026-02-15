# Algorithmic Market Prediction Platform

A fast, explainable algorithmic prediction system for crypto and equity markets. Combines technical analysis, NLP-based news sentiment, earnings event data, and LLM reasoning into structured predictions with full transparency into _why_ each prediction was made.

Built as a research project exploring how to fuse diverse market signals (quantitative, textual, event-driven) into a unified prediction pipeline with strict latency budgets and human-readable rationale.

## What it does

- **Multi-factor prediction engine**: Produces BUY/SELL/HOLD predictions with confidence scores, factor breakdowns, and natural-language rationale
- **Technical analysis**: RSI, EMA crossovers, Bollinger Bands, breakout detection, ATR-based volatility scoring
- **NLP sentiment**: Aggregates news from multiple APIs (NewsAPI, Finnhub, Alpha Vantage) and scores via VADER + TextBlob
- **Earnings events**: Monitors earnings reports and economic calendar for event-driven signals
- **LLM integration**: Uses Google Gemini (free tier) or OpenRouter for generating human-readable prediction rationale
- **Guardrails**: Confidence-weighting, volatility flags, and configurable exposure limits
- **Dashboard**: Streamlit web UI for live monitoring, charts, prediction logs, and explainability

## Prediction Output

Every prediction follows a structured schema:

```json
{
  "symbol": "BTC/USD",
  "timestamp": "2025-01-15T10:30:00Z",
  "action": "BUY",
  "confidence": 0.72,
  "horizon": "1h",
  "factor_breakdown": {
    "technical": 0.35,
    "sentiment": 0.22,
    "earnings": 0.0,
    "llm": 0.15
  },
  "rationale": "RSI oversold at 28 with bullish EMA crossover. Positive sentiment from 5 recent articles. No upcoming earnings events.",
  "risk_flags": ["high_volatility"],
  "latency_ms": 847.3
}
```

## Setup

```bash
# Create venv and install deps
python -m venv venv
venv\Scripts\activate        # windows
# source venv/bin/activate   # linux/mac

pip install -r requirements.txt

# Copy config template
cp config_example.py config.py
# Edit config.py with your Alpaca keys (required for live data connectivity)

# Verify setup
python setup_check.py
```

### LLM Setup (optional but recommended)

For prediction rationale generation, set one API key in your `.env` or `config.py`:

```bash
# Google Gemini (recommended - free tier, 15 RPM, fast)
GEMINI_API_KEY=your_gemini_key

# OR OpenRouter (alternative)
OPENROUTER_API_KEY=your_openrouter_key
```

**Why Gemini?** Google's Gemini API is ideal for this use case:

- Free tier with 15 requests/minute (sufficient for intraday prediction refresh)
- Strong reasoning capability for market analysis
- Fast response times (~500-1000ms)
- Structured output support

Without an LLM key, the engine still works but generates deterministic rationale from signal data alone.

## Running

**Dashboard UI** (recommended):

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

**Headless prediction runner** (CLI):

```bash
python run_prediction_engine.py

# With custom params:
python run_prediction_engine.py --symbols BTC/USD ETH/USD --capital 10000 --risk-percent 1.5
```

**Run tests**:

```bash
python tests/test_prediction.py    # prediction engine tests (21 tests)
python tests/test_suite.py         # full test suite
```

## Project Structure

```
project/
├── prediction/                    # Core prediction engine
│   ├── engine.py                 # Signal fusion + prediction pipeline
│   └── schema.py                 # Prediction dataclass (output contract)
├── dashboard/                     # Streamlit web interface
│   ├── app.py                    # Main dashboard
│   ├── pages/                    # Sub-pages (prediction, news, analytics)
│   └── components/               # Reusable UI components
├── strategy/                      # Signal strategy modules
│   ├── prediction_runtime_manager.py   # Main prediction loop (uses prediction engine)
│   ├── multiple_strategies.py    # Pluggable strategy framework
│   ├── news_strategy.py          # Sentiment-based signals
│   └── earnings_report_strategy.py # Event-driven signals
├── utils/                         # Shared utilities
│   ├── llm_integration.py        # Gemini / OpenRouter wrapper
│   ├── signal_processor.py       # Weighted signal fusion
│   ├── sentiment_analyzer.py     # Multi-source NLP sentiment
│   ├── websocket_manager.py      # Alpaca real-time data
│   ├── risk_management.py        # Prediction guardrail helpers
│   └── market_data_manager.py    # Historical data + caching
├── security/                      # Encrypted config storage
├── tests/                         # Test suite
├── docs/                          # Architecture, audit, cleanup docs
├── config.py                      # Main configuration (env vars)
├── run_prediction_engine.py      # Entry point (headless prediction mode)
├── run_bot.py                    # Entry point (dashboard)
└── README.md                      # This file
```

## How the Prediction Engine Works

The engine fuses signals from multiple sources using weighted combination:

1. **Technical analysis** (weight: 0.45) - RSI, EMA crossovers, Bollinger Bands, breakouts, ATR volatility
2. **NLP sentiment** (weight: 0.25) - News from 3 APIs, scored with VADER + TextBlob, weighted by source
3. **Earnings events** (weight: 0.15) - Upcoming earnings dates and economic releases
4. **LLM reasoning** (weight: 0.15) - Gemini/OpenRouter generates human-readable rationale

**Decision logic:**

- Combined confidence < 35%: HOLD (uncertain)
- Combined confidence ≥ 35%, positive signals: BUY
- Combined confidence ≥ 35%, negative signals: SELL

Each prediction includes:

- Factor-by-factor breakdown (why each signal contributed)
- Risk flags (high volatility, low coverage, earnings nearby, etc.)
- End-to-end latency (target: <1s without LLM, <4s with LLM)

### Extending with Custom Strategies

```python
from strategy.multiple_strategies import TradingStrategy

class MyStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("momentum_reversal", "mean reversion on momentum exhaustion")

    def generate_signals(self, df):
        # Your custom logic
        return {'signals': [...], 'metadata': {'confidence': 0.8}}
```

## Configuration

All settings via environment variables or `config.py`:

```python
# Required - Alpaca API keys (data/runtime access)
API_KEY = "your_alpaca_key"
API_SECRET = "your_alpaca_secret"

# Optional - news sentiment APIs
NEWS_API_KEY = "..."
FINNHUB_API_KEY = "..."
ALPHAVANTAGE_API_KEY = "..."

# Optional - LLM for rationale (pick one)
GEMINI_API_KEY = "..."   # Recommended (free)
OPENROUTER_API_KEY = "..."

# Runtime parameters
CAPITAL = 10000
RISK_PERCENT = 1.0
DEFAULT_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XAU/USD']
```

## Documentation

- [Architecture & Audit](docs/AUDIT_AND_ARCHITECTURE.md) - Current prediction-platform architecture and module classification
- [Cleanup Plan](docs/CLEANUP_PLAN.md) - Finalized cleanup actions and keep-list
- [Cleanup Report](docs/CLEANUP_REPORT.md) - Current removed vs kept inventory and verification results
- [API Reference](docs/API_REFERENCE.md) - Module and function documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Test Suite](tests/README.md) - Testing framework and coverage

## Recent Changes (Platform Reposition)

The project has been repositioned from a basic crypto execution bot into a **fast, explainable Algorithmic Market Prediction Platform**.

✅ **New prediction engine** - Structured Prediction schema with explainability (21 tests pass)
✅ **Upgraded LLM** - Gemini (primary) + OpenRouter (fallback) with intelligent fallback  
✅ **Cleaned codebase** - Removed legacy/dead paths and finalized duplicate-module cleanup  
✅ **Fixed 10 P0 bugs** - Syntax errors, undefined variables, broken imports  
✅ **Wired prediction_runtime_manager** - Now produces structured Predictions alongside legacy signals

See [AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md) for full technical details on audit, bugs, and architectural changes.

## Disclaimer

This is for educational and research purposes. Market predictions involve uncertainty. Validate outputs before any real-world use.

---

**Last updated:** February 2026  
**Status:** Production-ready (prediction mode). Requires API keys for live market connectivity.
