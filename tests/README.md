#  Testing Framework Documentation

## **Overview**

The trading bot includes a comprehensive test suite that validates core functionality across all modules. The testing framework ensures reliability, performance, and correctness of trading algorithms.

## **Test Structure**

### **Test Categories**

1. **Trading Strategies** (`TestTradingStrategies`)
   - Technical indicator calculations
   - Signal generation logic
   - Strategy performance validation

2. **Risk Management** (`TestRiskManagement`)
   - Position sizing algorithms
   - Stop loss calculations
   - Portfolio risk metrics

3. **Data Management** (`TestDataManager`)
   - Market data validation
   - API integration testing
   - Data standardization

4. **WebSocket Management** (`TestWebSocketManager`)
   - Real-time data streaming
   - Connection handling
   - Data format standardization

5. **Sentiment Analysis** (`TestSentimentAnalysis`)
   - News processing
   - Sentiment scoring
   - Signal generation

6. **Database Operations** (`TestDatabase`)
   - Trade storage/retrieval
   - Data integrity
   - Query performance

7. **Configuration** (`TestConfiguration`)
   - Config loading/validation
   - Environment handling
   - Security compliance

8. **API Integration** (`TestAPIIntegration`)
   - External API connectivity
   - Error handling
   - Rate limiting

## **Running Tests**

### **Complete Test Suite**
```bash
# Run all tests
python tests/test_suite.py

# Run with verbose output
python tests/test_suite.py -v
```

### **Individual Test Categories**
```bash
# Test specific module
python -m unittest tests.test_suite.TestTradingStrategies

# Test specific function
python -m unittest tests.test_suite.TestTradingStrategies.test_bollinger_bands_calculation
```

### **With Coverage Analysis**
```bash
# Install coverage tool
pip install coverage

# Run tests with coverage
coverage run tests/test_suite.py
coverage report
coverage html  # Generate HTML report
```

## **Test Implementation**

### **Sample Test Structure**
```python
class TestTradingStrategies(unittest.TestCase):
    """Test trading strategy functionality"""
    
    def setUp(self):
        """Set up test data before each test"""
        # Create sample market data
        self.sample_data = self._generate_sample_data()
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        result = calculate_bollinger_bands(self.sample_data['close'])
        
        # Validate return format
        self.assertIsInstance(result, dict)
        self.assertIn('upper', result)
        self.assertIn('middle', result)
        self.assertIn('lower', result)
        
        # Validate mathematical relationships
        valid_idx = ~pd.isna(result['upper'])
        if valid_idx.any():
            self.assertTrue((result['upper'][valid_idx] >= result['middle'][valid_idx]).all())
            self.assertTrue((result['middle'][valid_idx] >= result['lower'][valid_idx]).all())
    
    def tearDown(self):
        """Clean up after each test"""
        # Clean up any test artifacts
        pass
```

### **Mock Usage Examples**
```python
@patch('utils.market_data_manager.alpaca_trade_api.REST')
def test_market_data_fetching(self, mock_alpaca):
    """Test market data fetching with mocked API"""
    # Configure mock response
    mock_client = Mock()
    mock_alpaca.return_value = mock_client
    
    mock_response = Mock()
    mock_response.df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000] * 10
    })
    mock_client.get_crypto_bars.return_value = mock_response
    
    # Test the function
    manager = MarketDataManager({'data_source': 'alpaca'})
    result = manager.get_historical_data('BTC/USD', '1h', 10)
    
    # Validate results
    self.assertIsInstance(result, pd.DataFrame)
    self.assertEqual(len(result), 10)
    self.assertIn('close', result.columns)
```

## **Testing Best Practices**

### **1. Test Data Management**
- Use consistent seed values for reproducible results
- Generate realistic market data patterns
- Test edge cases (empty data, extreme values)

### **2. Mocking External Dependencies**
- Mock all API calls to external services
- Simulate network failures and timeouts
- Test rate limiting scenarios

### **3. Assertion Strategies**
- Validate data types and shapes
- Check mathematical relationships
- Verify error handling

### **4. Test Isolation**
- Each test should be independent
- Clean up resources in tearDown()
- Avoid test interdependencies

## **Test Data Generation**

### **Market Data Simulation**
```python
def generate_realistic_ohlcv(periods=100, base_price=50000, volatility=0.02):
    """Generate realistic OHLCV data for testing"""
    np.random.seed(42)  # Reproducible results
    
    # Generate price movements with realistic characteristics
    returns = np.random.normal(0, volatility, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, periods))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, periods))),
        'close': prices,
        'volume': np.random.randint(100, 10000, periods)
    })
    
    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df
```

### **News Data Simulation**
```python
def generate_sample_news(symbol='BTC', sentiment='positive', count=5):
    """Generate sample news data for sentiment testing"""
    positive_templates = [
        f"{symbol} reaches new all-time high amid institutional adoption",
        f"Major corporation announces {symbol} integration",
        f"{symbol} shows strong bullish momentum in technical analysis"
    ]
    
    negative_templates = [
        f"{symbol} faces regulatory concerns in major market",
        f"Technical analysis shows bearish divergence for {symbol}",
        f"{symbol} drops on profit-taking and market uncertainty"
    ]
    
    templates = positive_templates if sentiment == 'positive' else negative_templates
    
    return [
        {
            'title': template,
            'summary': f"Detailed analysis about {template.lower()}",
            'published_at': datetime.now().isoformat(),
            'source': f"NewsSource{i}",
            'url': f"https://example.com/news/{i}"
        }
        for i, template in enumerate(templates[:count])
    ]
```

## **Performance Testing**

### **Benchmark Framework**
```python
import time
import cProfile
import pstats

class PerformanceTester:
    """Performance testing utilities"""
    
    @staticmethod
    def benchmark_function(func, *args, **kwargs):
        """Benchmark a function's execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'function_name': func.__name__
        }
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """Profile a function's performance"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return result, stats

# Usage example
def test_strategy_performance(self):
    """Test strategy calculation performance"""
    large_dataset = self.generate_realistic_ohlcv(10000)
    
    benchmark = PerformanceTester.benchmark_function(
        calculate_bollinger_bands,
        large_dataset['close']
    )
    
    # Assert performance requirements
    self.assertLess(benchmark['execution_time'], 1.0)  # Should complete in < 1 second
```

## **Integration Testing**

### **End-to-End Test Scenarios**
```python
class TestTradingWorkflow(unittest.TestCase):
    """Test complete trading workflows"""
    
    def test_complete_trading_cycle(self):
        """Test full trading cycle from signal to execution"""
        # 1. Generate market data
        market_data = self.generate_test_data()
        
        # 2. Calculate indicators
        indicators = calculate_all_indicators(market_data)
        
        # 3. Generate signals
        signal = generate_trading_signal(indicators)
        
        # 4. Calculate position size
        position_size = calculate_position_size(
            capital=10000,
            risk_percent=1.0,
            signal=signal
        )
        
        # 5. Validate end-to-end results
        self.assertIn(signal['action'], ['buy', 'sell', 'hold'])
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 10000)
```

## **Continuous Integration**

### **GitHub Actions Workflow**
```yaml
name: Trading Bot Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install coverage pytest
    
    - name: Run tests with coverage
      run: |
        coverage run -m pytest tests/
        coverage xml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## **Test Maintenance**

### **Regular Tasks**

#### **Weekly**
- Review test coverage reports
- Update test data with recent market patterns
- Check for deprecated testing dependencies

#### **Monthly**
- Performance baseline updates
- Integration test validation with live APIs
- Test environment consistency checks

#### **Quarterly**
- Comprehensive test suite review
- Strategy backtesting validation
- Load testing for scalability

### **Test Evolution**

1. **Add Tests for New Features**
   - Write tests before implementing new functionality
   - Ensure comprehensive coverage of edge cases
   - Document test scenarios and expectations

2. **Refactor Tests with Code Changes**
   - Update tests when modifying existing functionality
   - Maintain test naming consistency
   - Keep test documentation current

3. **Monitor Test Performance**
   - Track test execution times
   - Optimize slow-running tests
   - Parallelize independent test suites

## **Debugging Failed Tests**

### **Common Failure Patterns**

1. **Data Type Mismatches**
   ```python
   # Expected: dict, Got: tuple
   if isinstance(result, tuple) and len(result) == 3:
       result = {'upper': result[0], 'middle': result[1], 'lower': result[2]}
   ```

2. **NaN Value Handling**
   ```python
   # Filter out NaN values before assertions
   valid_data = data.dropna()
   self.assertTrue((valid_data >= 0).all())
   ```

3. **Floating Point Precision**
   ```python
   # Use approximate equality for float comparisons
   self.assertAlmostEqual(result, expected, places=2)
   ```

### **Debugging Tools**

```python
# Add debugging output to failing tests
def test_with_debug(self):
    result = function_under_test(self.test_data)
    
    # Debug output
    print(f"Result type: {type(result)}")
    print(f"Result shape: {getattr(result, 'shape', 'N/A')}")
    print(f"Sample values: {result[:5] if hasattr(result, '__getitem__') else result}")
    
    # Continue with assertions
    self.assertIsInstance(result, expected_type)
```

---

This testing framework ensures the trading bot maintains high reliability and performance standards while facilitating continuous development and improvement.
