import logging
import os
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategy.earnings_report_strategy import EarningsReportStrategy
from strategy.news_strategy import NewsBasedStrategy
from strategy.strategy import check_entry
from utils.telegram_alert import send_alert
from prediction.engine import PredictionEngine
from prediction.schema import Prediction
from config import (
    API_KEY,
    API_SECRET,
    DAILY_PROFIT_TARGET_PERCENT,
    DEFAULT_SYMBOLS,
    PROFIT_TARGET_PERCENT,
    RISK_PERCENT,
    SENTIMENT_REFRESH_INTERVAL,
)

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None


class AutoTradingManager:
    """Manager for automated trading across multiple symbols."""

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
    ):
        self.logger = logging.getLogger("auto_trading_manager")
        self.setup_logging()

        self.symbols = [self._normalize_symbol(s) for s in (symbols or DEFAULT_SYMBOLS)]
        self.timeframe = self._normalize_timeframe(timeframe)
        self.capital = float(capital)
        self.risk_percent = float(risk_percent if risk_percent is not None else RISK_PERCENT)
        self.profit_target_percent = float(
            profit_target_percent
            if profit_target_percent is not None
            else PROFIT_TARGET_PERCENT
        )
        self.daily_profit_target = float(
            daily_profit_target
            if daily_profit_target is not None
            else DAILY_PROFIT_TARGET_PERCENT
        )

        self.use_news = bool(use_news)
        self.news_weight = float(news_weight)
        self.use_earnings = bool(use_earnings)
        self.earnings_weight = float(earnings_weight)
        self.signal_only = bool(signal_only)

        self.running = False
        self.dataframes = {}
        self.active_trades = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.last_signal = {}
        self.symbols_trading_halted = {}

        self.sentiment_signals = {}
        self.earnings_signals = {}
        self.last_sentiment_refresh = {}
        self.pending_event_symbols = set()

        self.news_strategy = NewsBasedStrategy() if self.use_news else None
        self.earnings_strategy = EarningsReportStrategy() if self.use_earnings else None

        # Prediction engine for structured, explainable predictions
        self.prediction_engine = PredictionEngine(config={
            "signal_weights": {
                "technical": max(0.0, 1.0 - self.news_weight - self.earnings_weight),
                "sentiment": self.news_weight if self.use_news else 0.0,
                "earnings": self.earnings_weight if self.use_earnings else 0.0,
                "llm": 0.1,
            }
        })
        self.latest_predictions = {}  # symbol -> Prediction

        self.api = None
        self.client = None
        self._init_alpaca_client()

        self.logger.info(
            "Auto trading manager initialized | symbols=%s timeframe=%s signal_only=%s",
            self.symbols,
            self.timeframe,
            self.signal_only,
        )

    def setup_logging(self):
        """Setup logger handlers once."""
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return

        os.makedirs("logs", exist_ok=True)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            f"logs/auto_trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_alpaca_client(self):
        """Initialize Alpaca REST client if dependency and keys are available."""
        if tradeapi is None:
            self.logger.warning("alpaca_trade_api not installed; running with synthetic/mock data")
            return

        if not API_KEY or not API_SECRET:
            self.logger.warning("Alpaca API credentials missing; running with synthetic/mock data")
            return

        base_urls = [
            "https://paper-api.alpaca.markets",
            "https://api.alpaca.markets",
        ]

        for base_url in base_urls:
            try:
                candidate = tradeapi.REST(API_KEY, API_SECRET, base_url=base_url)
                candidate.get_account()
                self.api = candidate
                self.client = candidate
                self.logger.info("Connected to Alpaca API (%s)", base_url)
                return
            except Exception as exc:
                self.logger.warning("Failed Alpaca connection (%s): %s", base_url, exc)

        self.logger.warning("All Alpaca connection attempts failed; using synthetic/mock data")

    def _normalize_symbol(self, symbol):
        symbol = str(symbol).strip().upper()
        if "/" in symbol:
            return symbol
        if symbol.endswith("USD") and len(symbol) > 3:
            return f"{symbol[:-3]}/USD"
        return symbol

    def _normalize_timeframe(self, timeframe):
        tf = str(timeframe).strip()
        mapping = {
            "1m": "1Min",
            "3m": "3Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "2h": "2Hour",
            "4h": "4Hour",
            "6h": "6Hour",
            "12h": "12Hour",
            "1d": "1Day",
            "1min": "1Min",
            "3min": "3Min",
            "5min": "5Min",
            "15min": "15Min",
            "30min": "30Min",
            "1hour": "1Hour",
            "2hour": "2Hour",
            "4hour": "4Hour",
            "6hour": "6Hour",
            "12hour": "12Hour",
            "1day": "1Day",
        }
        return mapping.get(tf.lower(), tf)

    def run(self):
        """Run the auto trading manager in an infinite loop."""
        self.running = True
        self.logger.info("Starting auto trading manager")

        for symbol in self.symbols:
            self._load_historical_data(symbol)

        while self.running:
            try:
                self.run_cycle()
            except Exception as exc:
                self.logger.error("Error in auto trading cycle: %s", exc)
                traceback.print_exc()

            if self.running:
                time.sleep(60)

        self.logger.info("Auto trading manager stopped")

    def run_cycle(self):
        """Process one deterministic trading cycle for all symbols."""
        cycle_results = {}

        for symbol in self.symbols:
            if self.symbols_trading_halted.get(symbol, False):
                self.logger.warning("Trading for %s is halted, skipping", symbol)
                continue

            try:
                updated = self._update_candles(symbol)
                if not updated:
                    self.logger.warning("Could not update candles for %s", symbol)
                    continue

                self._calculate_indicators(symbol)

                if self.use_news:
                    self._update_sentiment(symbol)
                if self.use_earnings:
                    self._process_earnings_signals(symbol)

                cycle_results[symbol] = self._analyze_symbol(symbol)
            except Exception as exc:
                self.logger.error("Cycle error for %s: %s", symbol, exc)
                traceback.print_exc()

        return cycle_results

    def stop(self):
        """Stop the auto trading manager."""
        self.running = False
        self.logger.info("Stopping auto trading manager")

    def _fetch_alpaca_crypto_bars(self, symbol, days_back=7):
        """Fetch crypto bars from Alpaca and return a normalized dataframe."""
        if self.client is None:
            return None

        api_symbol = symbol.replace("/", "")
        end = datetime.now()
        start = end - timedelta(days=days_back)

        bars = self.client.get_crypto_bars(
            api_symbol,
            self.timeframe,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        ).df

        if bars is None or bars.empty:
            return None

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
        elif bars.index.name:
            bars = bars.reset_index()

        ts_col = "timestamp" if "timestamp" in bars.columns else bars.columns[0]
        normalized = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(bars[ts_col]),
                "open": bars["open"].astype(float),
                "high": bars["high"].astype(float),
                "low": bars["low"].astype(float),
                "close": bars["close"].astype(float),
                "volume": bars["volume"].astype(float),
            }
        )
        normalized = normalized.set_index("timestamp").sort_index()
        return normalized

    def _load_historical_data(self, symbol):
        """Load historical data for a symbol and populate self.dataframes."""
        try:
            df = self._fetch_alpaca_crypto_bars(symbol, days_back=7)
            if df is None or df.empty:
                df = self._generate_mock_data(symbol)

            if df is None or df.empty:
                self.symbols_trading_halted[symbol] = True
                self.logger.error("Failed to load any data for %s; halting symbol", symbol)
                return False

            self.dataframes[symbol] = df
            self.symbols_trading_halted[symbol] = False
            return True
        except Exception as exc:
            self.logger.error("Error loading historical data for %s: %s", symbol, exc)
            self.symbols_trading_halted[symbol] = True
            return False

    def _generate_mock_data(self, symbol):
        """Generate mock price data for testing when API fails."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            num_candles = 168
            dates = pd.date_range(start=start_date, end=end_date, periods=num_candles)

            if "BTC" in symbol:
                base_price = 65000
            elif "ETH" in symbol:
                base_price = 3500
            elif "BNB" in symbol:
                base_price = 600
            elif "SOL" in symbol:
                base_price = 150
            elif "ADA" in symbol:
                base_price = 0.5
            elif "DOGE" in symbol:
                base_price = 0.15
            elif "XAU" in symbol or "GOLD" in symbol:
                base_price = 2400
            else:
                base_price = 100

            np.random.seed(42)
            price_changes = np.random.normal(0, base_price * 0.01, num_candles)
            prices = base_price + np.cumsum(price_changes)
            prices = np.maximum(prices, base_price * 0.5)

            mock_data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": prices * (1 - 0.005 * np.random.random(num_candles)),
                    "high": prices * (1 + 0.01 * np.random.random(num_candles)),
                    "low": prices * (1 - 0.01 * np.random.random(num_candles)),
                    "close": prices,
                    "volume": np.random.randint(100, 10000, num_candles),
                }
            )
            mock_data = mock_data.set_index("timestamp")
            return mock_data
        except Exception as exc:
            self.logger.error("Error generating mock data for %s: %s", symbol, exc)
            return None

    def _update_candles(self, symbol):
        """Update candle data for a specific symbol."""
        if symbol not in self.dataframes or self.dataframes[symbol] is None:
            return self._load_historical_data(symbol)

        try:
            latest_df = self._fetch_alpaca_crypto_bars(symbol, days_back=1)
            if latest_df is None or latest_df.empty:
                return True

            merged = pd.concat([self.dataframes[symbol], latest_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if len(merged) > 1500:
                merged = merged.iloc[-1000:]
            self.dataframes[symbol] = merged
            return True
        except Exception as exc:
            self.logger.error("Error updating candles for %s: %s", symbol, exc)
            return False

    def _calculate_indicators(self, symbol):
        """Calculate technical indicators for a symbol."""
        df = self.dataframes.get(symbol)
        if df is None or len(df) < 30:
            return

        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["sma20"] = close.rolling(window=20).mean()
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
        df["ema200"] = close.ewm(span=200, adjust=False).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        middle_band = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        df["middle_band"] = middle_band
        df["upper_band"] = middle_band + (std * 2)
        df["lower_band"] = middle_band - (std * 2)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        self.dataframes[symbol] = df

    def _update_sentiment(self, symbol):
        """Refresh sentiment signal for a symbol with safe neutral fallback."""
        if not self.news_strategy:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "news disabled"}

        now = time.time()
        last_refresh = self.last_sentiment_refresh.get(symbol, 0)
        event_forced = symbol in self.pending_event_symbols
        stale = now - last_refresh >= SENTIMENT_REFRESH_INTERVAL

        if symbol in self.sentiment_signals and not stale and not event_forced:
            return self.sentiment_signals[symbol]

        try:
            signal = self.news_strategy.generate_signal(symbol.replace("/", ""))
            normalized = {
                "signal": str(signal.get("signal", "neutral")).lower(),
                "confidence": float(signal.get("confidence", 0.0)),
                "reasoning": signal.get("reasoning", "news sentiment"),
                "raw": signal,
            }
        except Exception as exc:
            self.logger.warning("Sentiment update failed for %s: %s", symbol, exc)
            normalized = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": f"sentiment error: {exc}",
                "raw": {},
            }

        self.sentiment_signals[symbol] = normalized
        self.last_sentiment_refresh[symbol] = now
        return normalized

    def _process_earnings_signals(self, symbol):
        """Refresh earnings signal for a symbol with safe neutral fallback."""
        if not self.earnings_strategy:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "earnings disabled"}

        df = self.dataframes.get(symbol)
        event_forced = symbol in self.pending_event_symbols
        if symbol in self.earnings_signals and not event_forced:
            return self.earnings_signals[symbol]

        try:
            signal = self.earnings_strategy.generate_signal(
                symbol.replace("/", ""),
                data=df.reset_index() if df is not None else None,
            )
            normalized = {
                "signal": str(signal.get("signal", "neutral")).lower(),
                "confidence": float(signal.get("confidence", 0.0)),
                "reasoning": signal.get("reasoning", "earnings analysis"),
                "raw": signal,
            }
        except Exception as exc:
            self.logger.warning("Earnings signal failed for %s: %s", symbol, exc)
            normalized = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": f"earnings error: {exc}",
                "raw": {},
            }

        self.earnings_signals[symbol] = normalized
        return normalized

    def _build_technical_signal(self, symbol):
        """Build technical signal from current symbol dataframe."""
        df = self.dataframes.get(symbol)
        if df is None or len(df) < 2:
            return {"signal": "neutral", "confidence": 0.0, "reasoning": "insufficient data"}

        try:
            signal, confidence = check_entry(df)
            normalized_signal = str(signal).lower()
            if normalized_signal == "buy":
                final_signal = "buy"
            elif normalized_signal == "sell":
                final_signal = "sell"
            else:
                final_signal = "neutral"
            return {
                "signal": final_signal,
                "confidence": float(confidence or 0.0),
                "reasoning": "technical strategy",
            }
        except Exception as exc:
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": f"technical error: {exc}",
            }

    def _signal_to_direction(self, signal):
        if signal == "buy":
            return 1.0
        if signal == "sell":
            return -1.0
        return 0.0

    def _combine_signals(self, technical_signal, sentiment_signal, earnings_signal):
        """Combine technical, sentiment, and earnings signals with weights."""
        components = []

        tech_weight = max(
            0.0,
            1.0
            - (self.news_weight if self.use_news else 0.0)
            - (self.earnings_weight if self.use_earnings else 0.0),
        )
        if tech_weight == 0.0 and (not self.use_news and not self.use_earnings):
            tech_weight = 1.0

        components.append(("technical", technical_signal, tech_weight))
        if self.use_news:
            components.append(("sentiment", sentiment_signal, max(0.0, self.news_weight)))
        if self.use_earnings:
            components.append(("earnings", earnings_signal, max(0.0, self.earnings_weight)))

        total_weight = sum(weight for _, _, weight in components)
        if total_weight <= 0:
            equal_weight = 1.0 / len(components)
            components = [(name, signal, equal_weight) for name, signal, _ in components]
            total_weight = 1.0

        weighted_score = 0.0
        reasoning = []
        for name, signal, weight in components:
            direction = self._signal_to_direction(signal.get("signal", "neutral"))
            confidence = float(signal.get("confidence", 0.0))
            weighted_score += direction * confidence * weight
            reasoning.append(
                f"{name}: {signal.get('signal', 'neutral')}@{confidence:.2f} (w={weight:.2f})"
            )

        normalized_score = weighted_score / total_weight
        abs_score = abs(normalized_score)
        if normalized_score >= 0.15:
            final_signal = "buy"
        elif normalized_score <= -0.15:
            final_signal = "sell"
        else:
            final_signal = "neutral"

        return {
            "signal": final_signal,
            "confidence": min(1.0, abs_score),
            "reasoning": "; ".join(reasoning),
            "score": normalized_score,
        }

    def _analyze_symbol(self, symbol):
        """Analyze symbol and produce a structured Prediction alongside legacy signal."""
        technical = self._build_technical_signal(symbol)
        sentiment = self.sentiment_signals.get(
            symbol, {"signal": "neutral", "confidence": 0.0, "reasoning": "no sentiment"}
        )
        earnings = self.earnings_signals.get(
            symbol, {"signal": "neutral", "confidence": 0.0, "reasoning": "no earnings"}
        )

        combined = self._combine_signals(technical, sentiment, earnings)

        # Produce structured prediction via the engine
        df = self.dataframes.get(symbol)
        prediction = self.prediction_engine.predict(
            symbol,
            df,
            sentiment_override=sentiment,
            earnings_override=earnings,
            horizon=self._horizon_from_timeframe(),
        )
        self.latest_predictions[symbol] = prediction
        self.logger.info("Prediction: %s", prediction.summary())

        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "technical": technical,
            "sentiment": sentiment,
            "earnings": earnings,
            "combined": combined,
            "prediction": prediction.to_dict(),
        }
        self.last_signal[symbol] = result

        signal = combined["signal"]
        confidence = combined["confidence"]
        if signal in ("buy", "sell") and confidence >= 0.55:
            message = (
                f"{symbol} signal: {signal.upper()} "
                f"(confidence={confidence:.2f}) | {combined['reasoning']}"
            )
            self.logger.info(message)
            try:
                send_alert(message)
            except Exception as exc:
                self.logger.warning("Failed to send alert for %s: %s", symbol, exc)

            if self.signal_only:
                self.logger.info("Signal-only mode enabled. No order placement for %s.", symbol)

        self.pending_event_symbols.discard(symbol)
        return result

    def _extract_payload_symbols(self, payload):
        """Extract symbol list from webhook payload."""
        symbols = []
        if not isinstance(payload, dict):
            return symbols

        if isinstance(payload.get("symbol"), str):
            symbols.append(payload["symbol"])

        raw_symbols = payload.get("symbols")
        if isinstance(raw_symbols, list):
            symbols.extend([s for s in raw_symbols if isinstance(s, str)])

        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("symbol"), str):
            symbols.append(data["symbol"])

        # Finnhub earnings payloads can carry ticker-like fields.
        for field in ("ticker", "code"):
            if isinstance(payload.get(field), str):
                symbols.append(payload[field])

        cleaned = []
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            if normalized in self.symbols:
                cleaned.append(normalized)
        return list(dict.fromkeys(cleaned))

    def process_news_event(self, payload):
        """Webhook callback for news events."""
        try:
            target_symbols = self._extract_payload_symbols(payload) or self.symbols
            for symbol in target_symbols:
                self.pending_event_symbols.add(symbol)
                self._update_sentiment(symbol)
                self._analyze_symbol(symbol)
            return True
        except Exception as exc:
            self.logger.error("Error processing news event: %s", exc)
            return False

    def process_earnings_event(self, payload):
        """Webhook callback for earnings events."""
        try:
            target_symbols = self._extract_payload_symbols(payload) or self.symbols
            for symbol in target_symbols:
                self.pending_event_symbols.add(symbol)
                self._process_earnings_signals(symbol)
                if self.use_news:
                    self._update_sentiment(symbol)
                self._analyze_symbol(symbol)
            return True
        except Exception as exc:
            self.logger.error("Error processing earnings event: %s", exc)
            return False

    def get_historical_data(self, symbol, timeframe, days_back=30):
        """Get historical data for compatibility with existing callers."""
        normalized_symbol = self._normalize_symbol(symbol)
        previous_timeframe = self.timeframe
        self.timeframe = self._normalize_timeframe(timeframe)
        try:
            df = self._fetch_alpaca_crypto_bars(normalized_symbol, days_back=days_back)
            if df is None or df.empty:
                df = self._generate_synthetic_data(normalized_symbol, days_back=days_back)
            if df is None:
                return None
            out = df.reset_index().rename(columns={"timestamp": "time"})
            return out[["time", "open", "high", "low", "close", "volume"]]
        finally:
            self.timeframe = previous_timeframe

    def _generate_synthetic_data(self, symbol, days_back=30):
        """Generate synthetic OHLCV data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            dates = pd.date_range(start=start_date, end=end_date, freq="1h")

            if "BTC" in symbol:
                base_price = 30000
            elif "ETH" in symbol:
                base_price = 2000
            elif "BNB" in symbol:
                base_price = 500
            elif "ADA" in symbol:
                base_price = 0.6
            elif "SOL" in symbol:
                base_price = 140
            elif "DOGE" in symbol:
                base_price = 0.15
            elif "XAU" in symbol or "GOLD" in symbol:
                base_price = 2400
            else:
                base_price = 100

            close_prices = [base_price]
            for _ in range(1, len(dates)):
                price_change = np.random.normal(0, base_price * 0.01)
                close_prices.append(max(0.01, close_prices[-1] + price_change))

            df = pd.DataFrame({"timestamp": dates, "close": close_prices})
            df["open"] = df["close"].shift(1)
            df.loc[df.index[0], "open"] = df.loc[df.index[0], "close"]
            df["high"] = df[["open", "close"]].max(axis=1) * (
                1 + np.abs(np.random.normal(0, 0.005, len(df)))
            )
            df["low"] = df[["open", "close"]].min(axis=1) * (
                1 - np.abs(np.random.normal(0, 0.005, len(df)))
            )
            df["volume"] = np.random.randint(100, 10000, len(df))
            return df.set_index("timestamp")
        except Exception as exc:
            self.logger.error("Error generating synthetic data for %s: %s", symbol, exc)
            return None

    def get_active_trades(self):
        """Get list of active trades."""
        return list(self.active_trades.values())

    def get_last_signal(self):
        """Get the last generated signals by symbol."""
        return self.last_signal

    def get_current_pnl(self):
        """Get current total PnL."""
        return self.total_pnl

    def get_latest_predictions(self):
        """Get the most recent Prediction object for each symbol."""
        return dict(self.latest_predictions)

    def _horizon_from_timeframe(self):
        """Map the configured timeframe to a prediction horizon string."""
        mapping = {
            "1Min": "1h", "3Min": "1h", "5Min": "1h", "15Min": "1h",
            "30Min": "1h", "1Hour": "1h", "2Hour": "4h", "4Hour": "4h",
            "6Hour": "4h", "12Hour": "1d", "1Day": "1d",
        }
        return mapping.get(self.timeframe, "1h")
