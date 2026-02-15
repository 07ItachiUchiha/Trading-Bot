"""
Prediction Engine - core of the Algorithmic Market Prediction Platform.

Combines technical analysis signals, NLP sentiment, earnings data, and
optional LLM reasoning into a single Prediction object with full
explainability.

Usage:
    engine = PredictionEngine(config)
    prediction = engine.predict("BTC/USD", price_df)
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from prediction.schema import Prediction

logger = logging.getLogger("prediction_engine")


class PredictionEngine:
    """Fuses multiple signal sources into an explainable market prediction."""

    # Default weights for each signal source
    DEFAULT_WEIGHTS = {
        "technical": 0.45,
        "sentiment": 0.25,
        "earnings": 0.15,
        "llm": 0.15,
    }

    # Confidence thresholds - below this we output HOLD
    MIN_CONFIDENCE_TO_ACT = 0.35

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.weights = dict(self.DEFAULT_WEIGHTS)

        # Allow weight overrides from config
        if "signal_weights" in self.config:
            for key, val in self.config["signal_weights"].items():
                if key in self.weights:
                    self.weights[key] = float(val)

        # Normalize weights so they sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        self.sentiment_analyzer = None
        self.news_strategy = None
        self.llm_client = None

        self._init_sentiment()
        self._init_llm()

        logger.info(
            "PredictionEngine initialized | weights=%s",
            {k: round(v, 3) for k, v in self.weights.items()},
        )

    def _init_sentiment(self):
        """Try to load the sentiment analyzer; skip gracefully if deps missing."""
        try:
            from utils.sentiment_analyzer import SentimentAnalyzer
            from config import (
                ALPHAVANTAGE_API_KEY,
                FINNHUB_API_KEY,
                NEWS_API_KEY,
            )

            api_keys = {
                "alphavantage": ALPHAVANTAGE_API_KEY,
                "finnhub": FINNHUB_API_KEY,
                "newsapi": NEWS_API_KEY,
            }
            # Only create if at least one key is present
            if any(api_keys.values()):
                self.sentiment_analyzer = SentimentAnalyzer(api_keys=api_keys)
                logger.info("Sentiment analyzer loaded")
            else:
                logger.info("No news API keys configured; sentiment disabled")
        except Exception as exc:
            logger.warning("Could not load sentiment analyzer: %s", exc)

    def _init_llm(self):
        """Try to load LLM integration for rationale generation."""
        try:
            from utils.llm_integration import get_llm_response

            self.llm_client = get_llm_response
            logger.info("LLM integration loaded")
        except Exception as exc:
            logger.warning("LLM integration unavailable: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sentiment_override: Optional[Dict] = None,
        earnings_override: Optional[Dict] = None,
        horizon: str = "1h",
    ) -> Prediction:
        """
        Produce a single prediction for *symbol*.

        Parameters
        ----------
        symbol : str
            Ticker/pair, e.g. "BTC/USD".
        price_data : pd.DataFrame
            OHLCV dataframe with columns: open, high, low, close, volume.
        sentiment_override : dict, optional
            Pre-computed sentiment signal dict with 'signal' and 'confidence'.
        earnings_override : dict, optional
            Pre-computed earnings signal dict.
        horizon : str
            Prediction horizon ('1h', '4h', '1d').

        Returns
        -------
        Prediction
        """
        t0 = time.perf_counter()
        factors = {}
        explanations = []
        risk_flags = []

        # 1. Technical analysis
        tech = self._analyze_technical(price_data)
        factors["technical"] = tech
        explanations.append(tech["explanation"])
        risk_flags.extend(tech.get("risk_flags", []))

        # 2. Sentiment / NLP
        sent = self._analyze_sentiment(symbol, sentiment_override)
        factors["sentiment"] = sent
        if sent["confidence"] > 0:
            explanations.append(sent["explanation"])
        risk_flags.extend(sent.get("risk_flags", []))

        # 3. Earnings
        earn = self._analyze_earnings(symbol, earnings_override)
        factors["earnings"] = earn
        if earn["confidence"] > 0:
            explanations.append(earn["explanation"])
        risk_flags.extend(earn.get("risk_flags", []))

        # 4. Fuse signals
        action, confidence, breakdown = self._fuse_signals(factors)

        # 5. LLM rationale (optional, non-blocking fallback)
        rationale = self._generate_rationale(
            symbol, action, confidence, factors, explanations
        )

        latency = (time.perf_counter() - t0) * 1000

        return Prediction(
            symbol=symbol,
            action=action,
            confidence=confidence,
            horizon=horizon,
            factor_breakdown=breakdown,
            rationale=rationale,
            risk_flags=list(set(risk_flags)),
            latency_ms=round(latency, 1),
        )

    # ------------------------------------------------------------------
    # Technical Analysis
    # ------------------------------------------------------------------

    def _analyze_technical(self, df: pd.DataFrame) -> Dict:
        """Score the current market conditions from price data."""
        result = {
            "signal": 0.0,    # -1 to +1  (negative = bearish)
            "confidence": 0.0,
            "explanation": "",
            "risk_flags": [],
        }

        if df is None or len(df) < 30:
            result["explanation"] = "Insufficient price history for technical analysis"
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]

        signals = []
        reasons = []

        # RSI
        rsi = self._compute_rsi(close, 14)
        if rsi is not None:
            if rsi > 70:
                signals.append(-0.6)
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                signals.append(0.6)
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 60:
                signals.append(-0.2)
                reasons.append(f"RSI leaning bearish ({rsi:.0f})")
            elif rsi < 40:
                signals.append(0.2)
                reasons.append(f"RSI leaning bullish ({rsi:.0f})")
            else:
                signals.append(0.0)

        # EMA crossover (12/50)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        if len(ema12) > 1 and len(ema50) > 1:
            current_diff = ema12.iloc[-1] - ema50.iloc[-1]
            prev_diff = ema12.iloc[-2] - ema50.iloc[-2]
            if current_diff > 0 and prev_diff <= 0:
                signals.append(0.7)
                reasons.append("Bullish EMA crossover (12/50)")
            elif current_diff < 0 and prev_diff >= 0:
                signals.append(-0.7)
                reasons.append("Bearish EMA crossover (12/50)")
            elif current_diff > 0:
                signals.append(0.15)
                reasons.append("Price above EMA50")
            else:
                signals.append(-0.15)
                reasons.append("Price below EMA50")

        # Bollinger Band position
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        if not pd.isna(sma20.iloc[-1]) and not pd.isna(std20.iloc[-1]) and std20.iloc[-1] > 0:
            upper = sma20.iloc[-1] + 2 * std20.iloc[-1]
            lower = sma20.iloc[-1] - 2 * std20.iloc[-1]
            price = close.iloc[-1]
            bb_pos = (price - lower) / (upper - lower)  # 0 = lower band, 1 = upper band
            if bb_pos > 0.95:
                signals.append(-0.5)
                reasons.append("Price touching upper Bollinger Band")
                result["risk_flags"].append("bollinger_squeeze")
            elif bb_pos < 0.05:
                signals.append(0.5)
                reasons.append("Price touching lower Bollinger Band")

        # ATR-based volatility flag
        atr = self._compute_atr(high, low, close, 14)
        if atr is not None and close.iloc[-1] > 0:
            atr_pct = atr / close.iloc[-1]
            if atr_pct > 0.05:
                result["risk_flags"].append("high_volatility")

        # Volume spike check
        if "volume" in df.columns:
            vol = df["volume"]
            avg_vol = vol.rolling(20).mean()
            if not pd.isna(avg_vol.iloc[-1]) and avg_vol.iloc[-1] > 0:
                vol_ratio = vol.iloc[-1] / avg_vol.iloc[-1]
                if vol_ratio > 2.5:
                    result["risk_flags"].append("volume_spike")
                    reasons.append(f"Volume spike ({vol_ratio:.1f}x average)")

        # Aggregate
        if signals:
            avg_signal = np.mean(signals)
            result["signal"] = np.clip(avg_signal, -1.0, 1.0)
            result["confidence"] = min(abs(avg_signal), 1.0)
        result["explanation"] = "; ".join(reasons) if reasons else "No strong technical signals"

        return result

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def _analyze_sentiment(self, symbol: str, override: Optional[Dict] = None) -> Dict:
        """Get NLP sentiment score from news sources."""
        result = {
            "signal": 0.0,
            "confidence": 0.0,
            "explanation": "",
            "risk_flags": [],
        }

        if override:
            sig_str = str(override.get("signal", "neutral")).lower()
            conf = float(override.get("confidence", 0.0))
            result["signal"] = {"buy": 0.6, "sell": -0.6, "bullish": 0.6, "bearish": -0.6}.get(
                sig_str, 0.0
            )
            result["confidence"] = conf
            result["explanation"] = override.get("reasoning", f"Sentiment: {sig_str}")
            return result

        if not self.sentiment_analyzer:
            result["explanation"] = "Sentiment analysis not available"
            return result

        try:
            clean_symbol = symbol.replace("/", "")
            sentiment = self.sentiment_analyzer.get_sentiment(clean_symbol)
            if sentiment:
                score = float(sentiment.get("overall_score", 0.0))
                result["signal"] = np.clip(score, -1.0, 1.0)
                result["confidence"] = min(abs(score), 1.0)
                article_count = sentiment.get("article_count", 0)
                result["explanation"] = (
                    f"Sentiment score {score:.2f} from {article_count} articles"
                )
                if article_count < 3:
                    result["risk_flags"].append("low_news_coverage")
        except Exception as exc:
            logger.warning("Sentiment analysis failed for %s: %s", symbol, exc)
            result["explanation"] = f"Sentiment error: {exc}"

        return result

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def _analyze_earnings(self, symbol: str, override: Optional[Dict] = None) -> Dict:
        """Check for upcoming earnings events that might affect price."""
        result = {
            "signal": 0.0,
            "confidence": 0.0,
            "explanation": "",
            "risk_flags": [],
        }

        if override:
            sig_str = str(override.get("signal", "neutral")).lower()
            conf = float(override.get("confidence", 0.0))
            mapping = {"buy": 0.5, "sell": -0.5, "bullish": 0.5, "bearish": -0.5}
            result["signal"] = mapping.get(sig_str, 0.0)
            result["confidence"] = conf
            result["explanation"] = override.get("reasoning", f"Earnings: {sig_str}")
            if conf > 0.3:
                result["risk_flags"].append("earnings_event_nearby")
            return result

        # Without an override there is nothing to add; the earnings strategy
        # is used upstream in auto_trading_manager and passed in as override
        return result

    # ------------------------------------------------------------------
    # Signal Fusion
    # ------------------------------------------------------------------

    def _fuse_signals(self, factors: Dict) -> tuple:
        """
        Weighted combination of all factor signals into a final action.

        Returns (action, confidence, breakdown_dict).
        """
        weighted_sum = 0.0
        breakdown = {}

        for source, weight in self.weights.items():
            factor = factors.get(source, {})
            sig = float(factor.get("signal", 0.0))
            conf = float(factor.get("confidence", 0.0))

            # Weight the signal by both the source weight and the source's own confidence
            contribution = sig * weight * max(conf, 0.1)
            weighted_sum += contribution
            breakdown[source] = round(sig * conf, 3)

        # Determine action
        confidence = min(abs(weighted_sum) * 2, 1.0)  # Scale up for display

        if confidence < self.MIN_CONFIDENCE_TO_ACT:
            action = "HOLD"
        elif weighted_sum > 0:
            action = "BUY"
        else:
            action = "SELL"

        return action, round(confidence, 3), breakdown

    # ------------------------------------------------------------------
    # LLM Rationale
    # ------------------------------------------------------------------

    def _generate_rationale(
        self,
        symbol: str,
        action: str,
        confidence: float,
        factors: Dict,
        explanations: list,
    ) -> str:
        """
        Ask the LLM to produce a short human-readable rationale.

        Falls back to a deterministic summary if LLM is unavailable.
        """
        # Build deterministic fallback first
        parts = [exp for exp in explanations if exp]
        deterministic = f"{action} {symbol} (confidence {confidence:.0%}). " + "; ".join(parts)

        if not self.llm_client:
            return deterministic

        try:
            prompt = (
                f"You are a market analyst. Given the following signals for {symbol}, "
                f"write a 2-3 sentence rationale for the {action} recommendation "
                f"(confidence {confidence:.0%}).\n\n"
                f"Signals:\n" + "\n".join(f"- {e}" for e in parts if e) + "\n\n"
                f"Be concise, factual, and avoid hype. Do not use emoji."
            )
            response = self.llm_client(prompt)
            if response and len(response.strip()) > 10:
                return response.strip()
        except Exception as exc:
            logger.warning("LLM rationale generation failed: %s", exc)

        return deterministic

    # ------------------------------------------------------------------
    # Helper computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
        if len(close) < period + 1:
            return None
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_series = tr.rolling(period).mean()
        val = atr_series.iloc[-1]
        return None if pd.isna(val) else float(val)
