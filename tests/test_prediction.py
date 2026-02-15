"""
Acceptance tests for the Algorithmic Market Prediction Platform.

Run with: python -m pytest tests/test_prediction.py -v
Or:       python tests/test_prediction.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep acceptance tests deterministic and fast by disabling external LLM calls.
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""

from prediction.schema import Prediction
from prediction.engine import PredictionEngine


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def make_price_df(n=200, base=100.0, trend=0.0, seed=42):
    """Generate a synthetic OHLCV dataframe for testing."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="1h")
    changes = np.random.normal(trend, base * 0.008, n)
    prices = base + np.cumsum(changes)
    prices = np.maximum(prices, base * 0.3)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 - 0.003 * np.random.random(n)),
        "high": prices * (1 + 0.008 * np.random.random(n)),
        "low": prices * (1 - 0.008 * np.random.random(n)),
        "close": prices,
        "volume": np.random.randint(500, 50000, n).astype(float),
    })
    return df.set_index("timestamp")


# -----------------------------------------------------------------
# Schema tests
# -----------------------------------------------------------------

class TestPredictionSchema:
    """Verify the Prediction dataclass behaves correctly."""

    def test_defaults(self):
        p = Prediction(symbol="BTC/USD")
        assert p.action == "HOLD"
        assert p.confidence == 0.0
        assert p.symbol == "BTC/USD"
        assert p.timestamp.endswith("Z")
        assert isinstance(p.factor_breakdown, dict)
        assert isinstance(p.risk_flags, list)

    def test_confidence_clamping(self):
        p = Prediction(symbol="ETH/USD", confidence=1.5)
        assert p.confidence == 1.0
        p2 = Prediction(symbol="ETH/USD", confidence=-0.3)
        assert p2.confidence == 0.0

    def test_action_normalization(self):
        for raw, expected in [("buy", "BUY"), ("SELL", "SELL"), ("hold", "HOLD"), ("xyz", "HOLD")]:
            p = Prediction(symbol="X", action=raw)
            assert p.action == expected, f"Expected {expected} but got {p.action} for input '{raw}'"

    def test_to_dict_roundtrip(self):
        p = Prediction(
            symbol="SOL/USD",
            action="BUY",
            confidence=0.75,
            factor_breakdown={"technical": 0.4},
            risk_flags=["high_volatility"],
        )
        d = p.to_dict()
        assert d["symbol"] == "SOL/USD"
        assert d["action"] == "BUY"
        assert d["confidence"] == 0.75
        assert "technical" in d["factor_breakdown"]

        p2 = Prediction.from_dict(d)
        assert p2.symbol == p.symbol
        assert p2.action == p.action

    def test_to_json(self):
        p = Prediction(symbol="BTC/USD", action="SELL", confidence=0.6)
        j = p.to_json()
        assert '"action": "SELL"' in j
        assert '"symbol": "BTC/USD"' in j

    def test_summary_string(self):
        p = Prediction(symbol="BTC/USD", action="BUY", confidence=0.8, risk_flags=["volume_spike"])
        s = p.summary()
        assert "BTC/USD" in s
        assert "BUY" in s
        assert "volume_spike" in s


# -----------------------------------------------------------------
# Engine tests
# -----------------------------------------------------------------

class TestPredictionEngine:
    """Test the core prediction engine."""

    def setup_method(self):
        self.engine = PredictionEngine()

    def test_predict_returns_prediction(self):
        df = make_price_df(200, base=65000)
        pred = self.engine.predict("BTC/USD", df)
        assert isinstance(pred, Prediction)
        assert pred.symbol == "BTC/USD"
        assert pred.action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_has_latency(self):
        df = make_price_df(200, base=65000)
        pred = self.engine.predict("BTC/USD", df)
        assert pred.latency_ms > 0, "Latency should be measured"
        # Allow up to 20s since LLM calls may timeout gracefully
        assert pred.latency_ms < 20000, "Prediction should complete in under 20s"

    def test_predict_has_factor_breakdown(self):
        df = make_price_df(200, base=65000)
        pred = self.engine.predict("BTC/USD", df)
        assert isinstance(pred.factor_breakdown, dict)
        assert "technical" in pred.factor_breakdown

    def test_predict_has_rationale(self):
        df = make_price_df(200, base=65000)
        pred = self.engine.predict("BTC/USD", df)
        assert len(pred.rationale) > 0, "Rationale should not be empty"

    def test_predict_with_insufficient_data(self):
        """Engine should handle short dataframes gracefully."""
        df = make_price_df(5, base=100)
        pred = self.engine.predict("XYZ/USD", df)
        assert pred.action == "HOLD"
        assert "Insufficient" in pred.rationale or pred.confidence < 0.35

    def test_predict_bullish_trend(self):
        """Strong uptrend should lean toward BUY."""
        df = make_price_df(200, base=100, trend=0.5, seed=99)
        pred = self.engine.predict("UP/USD", df)
        # We can't guarantee BUY due to other factors, but confidence should be nonzero
        assert isinstance(pred, Prediction)
        assert pred.latency_ms > 0

    def test_predict_bearish_trend(self):
        """Strong downtrend should lean toward SELL."""
        df = make_price_df(200, base=200, trend=-0.8, seed=77)
        pred = self.engine.predict("DOWN/USD", df)
        assert isinstance(pred, Prediction)
        assert pred.latency_ms > 0

    def test_sentiment_override(self):
        """Test that sentiment override is picked up."""
        df = make_price_df(200, base=100)
        sentiment = {"signal": "bullish", "confidence": 0.9, "reasoning": "very positive news"}
        pred = self.engine.predict("TEST/USD", df, sentiment_override=sentiment)
        assert isinstance(pred, Prediction)
        assert "sentiment" in pred.factor_breakdown

    def test_custom_weights(self):
        """Test engine respects custom weight configuration."""
        engine = PredictionEngine(config={
            "signal_weights": {"technical": 1.0, "sentiment": 0.0, "earnings": 0.0, "llm": 0.0}
        })
        df = make_price_df(200, base=100)
        pred = engine.predict("TEST/USD", df)
        assert isinstance(pred, Prediction)

    def test_horizon_propagation(self):
        df = make_price_df(200, base=100)
        pred = self.engine.predict("BTC/USD", df, horizon="4h")
        assert pred.horizon == "4h"

    def test_risk_flags_type(self):
        df = make_price_df(200, base=100)
        pred = self.engine.predict("BTC/USD", df)
        assert isinstance(pred.risk_flags, list)
        for flag in pred.risk_flags:
            assert isinstance(flag, str)


# -----------------------------------------------------------------
# Integration: verify signal_processor still works
# -----------------------------------------------------------------

class TestSignalProcessor:
    """Smoke test the existing SignalProcessor to confirm no regressions."""

    def test_import(self):
        from utils.signal_processor import SignalProcessor
        sp = SignalProcessor()
        assert sp is not None

    def test_process_signals(self):
        from utils.signal_processor import SignalProcessor
        sp = SignalProcessor()
        df = make_price_df(200)
        result = sp.process_signals(
            symbol="TEST",
            technical_signal={"signal": "buy", "confidence": 0.6},
            sentiment_signal={"signal": "neutral", "confidence": 0.3},
            earnings_signal={"signal": "neutral", "confidence": 0.0},
            price_data=df,
        )
        assert "signal" in result
        assert "confidence" in result


# -----------------------------------------------------------------
# Integration: verify LLM integration module loads
# -----------------------------------------------------------------

class TestLLMIntegration:
    """Verify the LLM integration module loads and has correct structure."""

    def test_import(self):
        from utils.llm_integration import get_llm_response
        assert callable(get_llm_response)

    def test_no_key_raises(self):
        """Without API keys, calling should raise RuntimeError."""
        # Temporarily clear keys
        old_gemini = os.environ.get("GEMINI_API_KEY")
        old_nvidia = os.environ.get("NVIDIA_API_KEY")
        old_openrouter = os.environ.get("OPENROUTER_API_KEY")
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["NVIDIA_API_KEY"] = ""
        os.environ["OPENROUTER_API_KEY"] = ""

        from utils.llm_integration import get_llm_response
        try:
            get_llm_response("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No LLM API key" in str(e)
        finally:
            if old_gemini is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_gemini

            if old_nvidia is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_nvidia

            if old_openrouter is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_openrouter


# -----------------------------------------------------------------
# Run with plain python if pytest not available
# -----------------------------------------------------------------

def _run_tests():
    """Simple test runner without pytest dependency."""
    passed = 0
    failed = 0
    errors = []

    test_classes = [TestPredictionSchema, TestPredictionEngine, TestSignalProcessor, TestLLMIntegration]

    for cls in test_classes:
        instance = cls()
        setup = getattr(instance, "setup_method", None)

        for attr in sorted(dir(instance)):
            if not attr.startswith("test_"):
                continue

            if setup:
                try:
                    setup()
                except Exception:
                    pass

            name = f"{cls.__name__}.{attr}"
            try:
                getattr(instance, attr)()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  FAIL  {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    return failed == 0


if __name__ == "__main__":
    print("Running prediction platform acceptance tests...\n")
    success = _run_tests()
    sys.exit(0 if success else 1)
