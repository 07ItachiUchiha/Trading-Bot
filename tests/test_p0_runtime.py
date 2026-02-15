import unittest
from unittest.mock import patch

from strategy.auto_trading_manager import AutoTradingManager


class TestP0Runtime(unittest.TestCase):
    def test_run_cycle_executes_and_updates_last_signal(self):
        manager = AutoTradingManager(
            symbols=["BTC/USD"],
            timeframe="1h",
            use_news=False,
            use_earnings=False,
            signal_only=True,
        )

        result = manager.run_cycle()
        self.assertIn("BTC/USD", result)
        self.assertIn("BTC/USD", manager.get_last_signal())
        self.assertIn("combined", manager.get_last_signal()["BTC/USD"])

    def test_webhook_handlers_tolerate_missing_or_partial_payloads(self):
        manager = AutoTradingManager(
            symbols=["BTC/USD"],
            timeframe="1h",
            use_news=False,
            use_earnings=False,
            signal_only=True,
        )

        self.assertTrue(manager.process_news_event(None))
        self.assertTrue(manager.process_news_event({}))
        self.assertTrue(manager.process_earnings_event({"data": {}}))

    def test_signal_only_mode_does_not_create_trades(self):
        manager = AutoTradingManager(
            symbols=["BTC/USD"],
            timeframe="1h",
            use_news=False,
            use_earnings=False,
            signal_only=True,
        )
        manager._load_historical_data("BTC/USD")

        with patch(
            "strategy.auto_trading_manager.send_alert",
            return_value=True,
        ) as send_alert_mock:
            with patch.object(
                manager,
                "_build_technical_signal",
                return_value={
                    "signal": "buy",
                    "confidence": 0.9,
                    "reasoning": "forced for test",
                },
            ):
                signal_data = manager._analyze_symbol("BTC/USD")

        self.assertEqual(signal_data["combined"]["signal"], "buy")
        self.assertEqual(len(manager.get_active_trades()), 0)
        send_alert_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
