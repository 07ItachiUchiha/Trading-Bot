"""Test suite covering strategies, risk management, data, websockets, sentiment, etc."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import json
import importlib.util
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TestTradingStrategies(unittest.TestCase):
    """Test trading strategy functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')  # Use lowercase 'h'
        np.random.seed(42)  # For reproducible tests
        
        self.sample_data = pd.DataFrame({
            'time': dates,
            'open': 50000 + np.random.randn(100) * 1000,
            'high': 50000 + np.random.randn(100) * 1000 + 500,
            'low': 50000 + np.random.randn(100) * 1000 - 500,
            'close': 50000 + np.random.randn(100) * 1000,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high >= low >= open/close
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            prices = [row['open'], row['close']]
            self.sample_data.at[i, 'high'] = max(max(prices), row['high'])
            self.sample_data.at[i, 'low'] = min(min(prices), row['low'])
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        try:
            from dashboard.components.trading import calculate_bollinger_bands
            
            result = calculate_bollinger_bands(self.sample_data['close'])
            
            # Check if result is tuple (upper, middle, lower) or dict
            if isinstance(result, tuple) and len(result) == 3:
                upper, middle, lower = result
                result = {'upper': upper, 'middle': middle, 'lower': lower}
            
            self.assertIsInstance(result, dict)
            self.assertIn('upper', result)
            self.assertIn('middle', result)
            self.assertIn('lower', result)
            
            # Test that upper > middle > lower (generally, ignoring NaN values)
            valid_idx = ~pd.isna(result['upper']) & ~pd.isna(result['middle']) & ~pd.isna(result['lower'])
            if valid_idx.any():
                self.assertTrue((result['upper'][valid_idx] >= result['middle'][valid_idx]).all())
                self.assertTrue((result['middle'][valid_idx] >= result['lower'][valid_idx]).all())            
        except ImportError:
            self.skipTest("Trading components not available")
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        try:
            from dashboard.components.trading import calculate_rsi
            
            result = calculate_rsi(self.sample_data['close'])
            
            self.assertIsInstance(result, pd.Series)
            # RSI should be between 0 and 100 (ignoring NaN values)
            valid_rsi = result.dropna()
            if len(valid_rsi) > 0:
                self.assertTrue((valid_rsi >= 0).all())
                self.assertTrue((valid_rsi <= 100).all())
            
        except ImportError:
            self.skipTest("Trading components not available")
    
    def test_ema_calculation(self):
        """Test EMA calculation"""
        try:
            from dashboard.components.trading import calculate_ema
            
            result = calculate_ema(self.sample_data['close'])
            
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.sample_data))
            
        except ImportError:
            self.skipTest("Trading components not available")

class TestRiskManagement(unittest.TestCase):
    """Test risk management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.account_balance = 10000
        self.entry_price = 50000
        self.stop_loss = 49000
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        try:
            from dashboard.components.risk_management import calculate_position_size
            
            result = calculate_position_size(
                account_balance=self.account_balance,
                confidence_score=0.7,
                risk_percentage=1.0
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('position_size', result)
            self.assertIn('percentage', result)
            self.assertIn('risk_level', result)
            
            # Position size should be reasonable
            self.assertGreater(result['position_size'], 0)
            self.assertLess(result['position_size'], self.account_balance)            
        except ImportError:
            self.skipTest("Risk management module not available")
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        try:
            from dashboard.components.risk_management import calculate_stop_loss
            
            result = calculate_stop_loss(
                entry_price=self.entry_price,
                asset_volatility=0.02,
                direction="buy"
            )
            
            # The function returns a float, not a dict
            self.assertIsInstance(result, float)
            
            # Stop loss should be below entry for buy orders
            self.assertLess(result, self.entry_price)
            
        except ImportError:
            self.skipTest("Risk management module not available")

class TestDataManager(unittest.TestCase):
    """Test market data management"""
    
    def test_data_validation(self):
        """Test data validation functions"""
        try:
            from utils.dataframe_utils import validate_ohlc_data
            
            # Valid data
            valid_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [103, 104, 105],
                'volume': [1000, 1100, 1200]
            })
            
            result = validate_ohlc_data(valid_data)
            self.assertTrue(result['is_valid'])            
        except ImportError:
            self.skipTest("Data utils not available")
    
    def test_market_data_fetching(self):
        """Test market data fetching with mocked API"""
        if importlib.util.find_spec("alpaca_trade_api") is None:
            self.skipTest("alpaca_trade_api not installed")

        try:
            from utils.market_data_manager import MarketDataManager
            
            # Mock API response
            mock_bars = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),  # Use lowercase 'h'
                'open': np.random.rand(10) * 100,
                'high': np.random.rand(10) * 100,
                'low': np.random.rand(10) * 100,
                'close': np.random.rand(10) * 100,
                'volume': np.random.randint(1000, 10000, 10)
            })
            
            with patch("utils.market_data_manager.tradeapi.REST") as mock_rest:
                mock_rest.return_value.get_crypto_bars.return_value.df = mock_bars

                # Create manager with required config
                config = {'data_source': 'alpaca', 'default_timeframe': '1h'}
                api_keys = {"alpaca_key": "k", "alpaca_secret": "s"}
                manager = MarketDataManager(api_keys=api_keys, config=config)
                result = manager.get_historical_data('BTC/USD', '1h', 10)
            
            self.assertIsInstance(result, pd.DataFrame)
            
        except ImportError:
            self.skipTest("Market data manager not available")

class TestWebSocketManager(unittest.TestCase):
    """Test WebSocket functionality"""
    
    def test_data_standardization(self):
        """Test WebSocket data standardization"""
        try:
            from utils.websocket_manager import WebSocketManager
            
            # Sample raw data
            raw_data = {
                'S': 'BTCUSD',
                'p': '50000.0',
                's': '0.1',
                't': '2024-01-01T00:00:00Z'
            }
            
            # Create WebSocket manager with test credentials
            manager = WebSocketManager('test_key', 'test_secret', 'wss://test.com')
            result = manager._standardize_data(raw_data, 'trade')
            
            self.assertIsInstance(result, dict)
            self.assertIn('symbol', result)
            self.assertIn('price', result)
            
        except ImportError:
            self.skipTest("WebSocket manager not available")

class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality"""
    
    def test_sentiment_scoring(self):
        """Test sentiment analysis scoring"""
        try:
            from utils.sentiment_analyzer import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer()
            
            # Test positive sentiment
            positive_news = {
                'title': 'Bitcoin reaches new all-time high',
                'description': 'Great news for crypto investors',
                'content': 'Bitcoin has surged to record levels',
                'pre_analyzed_sentiment': None  # Add required field
            }
            
            result = analyzer._analyze_text_sentiment(positive_news)
            
            self.assertIsInstance(result, dict)
            self.assertIn('compound', result)
            self.assertGreater(result['compound'], 0)  # Should be positive
            
        except ImportError:
            self.skipTest("Sentiment analyzer not available")

class TestDatabase(unittest.TestCase):
    """Test database functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.test_db_path = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
    
    def test_trade_storage(self):
        """Test trade storage and retrieval"""
        try:
            from dashboard.components.database import add_trade_to_db, get_trades_from_db
            
            # Mock trade data
            trade_data = {
                'symbol': 'BTC/USD',
                'direction': 'long',
                'entry_price': 50000.0,
                'size': 0.1,
                'stop_loss': 49000.0,
                'targets': [51000.0, 52000.0],
                'status': 'open'
            }
            
            # This is a simplified test - in practice you'd mock the database
            # For now, just test that the functions exist and have correct signatures
            self.assertTrue(callable(add_trade_to_db))
            self.assertTrue(callable(get_trades_from_db))
            
        except ImportError:
            self.skipTest("Database components not available")

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            import config
            
            # Test that required configuration exists
            required_attrs = ['CAPITAL', 'RISK_PERCENT', 'DEFAULT_SYMBOLS']
            
            for attr in required_attrs:
                self.assertTrue(hasattr(config, attr), f"Missing config attribute: {attr}")
            
            # Test reasonable values
            self.assertGreater(config.CAPITAL, 0)
            self.assertGreater(config.RISK_PERCENT, 0)
            self.assertLessEqual(config.RISK_PERCENT, 100)
            
        except ImportError:
            self.skipTest("Config module not available")

class TestAPIIntegration(unittest.TestCase):
    """Test API integration functionality"""
    
    @patch('requests.get')
    def test_news_api_integration(self, mock_get):
        """Test news API integration"""
        try:
            from utils.news_fetcher import NewsFetcher
            
            # Mock API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'articles': [
                    {
                        'title': 'Test News',
                        'description': 'Test Description',
                        'url': 'http://test.com',
                        'publishedAt': '2024-01-01T00:00:00Z'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            fetcher = NewsFetcher()
            result = fetcher.get_news_for_symbol('BTCUSD')  # Use correct method name
            
            self.assertIsInstance(result, list)
            
        except ImportError:
            self.skipTest("News fetcher not available")

def run_comprehensive_tests():
    """Run everything and print a summary."""
    
    # Create test suite
    test_classes = [
        TestTradingStrategies,
        TestRiskManagement,
        TestDataManager,
        TestWebSocketManager,
        TestSentimentAnalysis,
        TestDatabase,
        TestConfiguration,
        TestAPIIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
