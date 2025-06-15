# 🔧 Trading Bot API Reference

## **Core Classes**

### **AutoTradingManager**
Primary orchestrator for automated trading operations.

```python
class AutoTradingManager:
    def __init__(self, symbols, timeframe, capital=10000, 
                 risk_percent=1.0, profit_target_percent=3.0,
                 daily_profit_target=5.0, use_news=True, 
                 news_weight=0.5, use_earnings=True, 
                 earnings_weight=0.6)
```

#### **Methods**

##### `run()`
Starts the main trading loop.
- **Returns**: None
- **Raises**: ConnectionError if API unavailable

##### `stop()`
Gracefully stops trading operations.
- **Returns**: None

##### `get_historical_data(symbol, timeframe, days_back=30)`
Retrieves historical price data.
- **Parameters**:
  - `symbol` (str): Trading symbol (e.g., 'BTC/USD')
  - `timeframe` (str): Data frequency ('1H', '1D')
  - `days_back` (int): Historical period
- **Returns**: pandas.DataFrame with OHLCV data
- **Raises**: APIError if data unavailable

---

### **SentimentAnalyzer**
Processes news and social media for trading signals.

```python
class SentimentAnalyzer:
    def __init__(self, api_keys=None, config=None)
```

#### **Methods**

##### `analyze_sentiment(symbol, news_items=None, days_back=3)`
Analyzes sentiment for a trading symbol.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `news_items` (list): Optional news data
  - `days_back` (int): Historical news period
- **Returns**: 
  ```python
  {
      'signal': 'buy'|'sell'|'neutral',
      'confidence': float,  # 0.0-1.0
      'score': float,       # -1.0 to 1.0
      'reasoning': str,
      'news_count': int,
      'source_count': int
  }
  ```

##### `fetch_news(symbol, days_back=3)`
Retrieves news articles for analysis.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `days_back` (int): Historical period
- **Returns**: List of news articles
- **Raises**: APIError if news service unavailable

---

### **RiskManager**
Manages position sizing and risk controls.

```python
class RiskManager:
    def __init__(self, config=None)
```

#### **Methods**

##### `calculate_position_size(symbol, entry_price, stop_loss, confidence=0.7, account_value=None)`
Calculates optimal position size.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `entry_price` (float): Planned entry price
  - `stop_loss` (float): Stop loss level
  - `confidence` (float): Signal confidence (0.0-1.0)
  - `account_value` (float): Account size
- **Returns**: 
  ```python
  {
      'size': float,           # Position size
      'risk_amount': float,    # Dollar risk
      'max_loss_percent': float
  }
  ```

##### `validate_trade(symbol, size, direction, current_positions)`
Validates trade against risk limits.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `size` (float): Position size
  - `direction` (str): 'long' or 'short'
  - `current_positions` (dict): Current portfolio
- **Returns**: bool (True if trade allowed)

---

### **SignalProcessor**
Combines signals from multiple sources.

```python
class SignalProcessor:
    def __init__(self, config=None)
```

#### **Methods**

##### `process_signals(symbol, technical_signal=None, sentiment_signal=None, earnings_signal=None, price_data=None)`
Combines and weights trading signals.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `technical_signal` (dict): Technical analysis signal
  - `sentiment_signal` (dict): Sentiment analysis signal
  - `earnings_signal` (dict): Earnings-based signal
  - `price_data` (DataFrame): Current price data
- **Returns**:
  ```python
  {
      'symbol': str,
      'signal': 'buy'|'sell'|'neutral',
      'confidence': float,     # 0.0-1.0
      'reasoning': str,        # Human-readable explanation
      'timestamp': datetime
  }
  ```

---

### **WebSocketManager**
Manages real-time data streams.

```python
class WebSocketManager:
    def __init__(self, api_key, api_secret, base_url)
```

#### **Methods**

##### `subscribe(symbol, callback)`
Subscribes to real-time data for a symbol.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `callback` (function): Data handler function
- **Returns**: None

##### `unsubscribe(symbol, callback=None)`
Unsubscribes from symbol data.
- **Parameters**:
  - `symbol` (str): Trading symbol
  - `callback` (function): Optional specific callback
- **Returns**: None

---

## **Utility Functions**

### **Symbol Standardization**

```python
def standardize_symbol(symbol, destination='alpaca')
```
Converts symbols between different exchange formats.
- **Parameters**:
  - `symbol` (str): Input symbol
  - `destination` (str): Target format ('alpaca', 'binance', 'display')
- **Returns**: str (formatted symbol)

### **Position Sizing**

```python
def calculate_position_size(capital, risk_percent, entry_price, stop_loss, max_capital_per_trade=0.15)
```
Calculates position size based on risk parameters.
- **Parameters**:
  - `capital` (float): Available capital
  - `risk_percent` (float): Risk percentage per trade
  - `entry_price` (float): Entry price level
  - `stop_loss` (float): Stop loss level
  - `max_capital_per_trade` (float): Maximum capital allocation
- **Returns**: float (position size)

### **Trailing Stop Loss**

```python
def calculate_trailing_stop(current_price, direction, atr, profit_ticks, initial_stop)
```
Calculates dynamic trailing stop levels.
- **Parameters**:
  - `current_price` (float): Current market price
  - `direction` (str): 'long' or 'short'
  - `atr` (float): Average True Range
  - `profit_ticks` (float): Profitable movement
  - `initial_stop` (float): Initial stop loss
- **Returns**: float (new stop level)

---

## **Configuration Objects**

### **Trading Configuration**

```python
{
    'symbols': ['BTC/USD', 'ETH/USD'],
    'timeframe': '1H',
    'capital': 10000,
    'risk_percent': 1.0,
    'profit_target_percent': 3.0,
    'daily_profit_target': 5.0,
    'max_capital_per_trade': 0.15
}
```

### **Strategy Configuration**

```python
{
    'use_news': True,
    'news_weight': 0.5,
    'use_earnings': True,
    'earnings_weight': 0.6,
    'sentiment_threshold': 0.7,
    'technical_weight': 0.4
}
```

### **Risk Management Configuration**

```python
{
    'max_correlation': 0.7,
    'max_portfolio_risk': 5.0,
    'stop_loss_percent': 2.0,
    'take_profit_percent': 4.0,
    'trailing_stop_enabled': True
}
```

---

## **Error Handling**

### **Common Exceptions**

- **APIError**: API connection or rate limit issues
- **DataError**: Invalid or missing market data
- **ConfigError**: Configuration validation failures
- **RiskError**: Risk limit violations
- **ConnectionError**: Network connectivity issues

### **Error Response Format**

```python
{
    'error': True,
    'message': str,      # Human-readable error
    'code': str,         # Error code for programmatic handling
    'details': dict,     # Additional context
    'timestamp': datetime
}
```

---

## **Event System**

### **Event Types**

- **trade_executed**: Trade completed
- **signal_generated**: New trading signal
- **risk_alert**: Risk limit approached
- **connection_lost**: API disconnection
- **data_updated**: New market data received

### **Event Handler Example**

```python
def handle_trade_event(event_data):
    """
    Handle trade execution events
    
    Args:
        event_data (dict): Event information
            - type: 'trade_executed'
            - symbol: str
            - size: float
            - price: float
            - timestamp: datetime
    """
    pass
```

---

## **Data Structures**

### **Price Data (OHLCV)**

```python
DataFrame columns:
- timestamp: datetime64[ns]
- open: float64
- high: float64
- low: float64
- close: float64
- volume: float64
```

### **Trade Record**

```python
{
    'id': str,
    'symbol': str,
    'side': 'buy'|'sell',
    'size': float,
    'price': float,
    'timestamp': datetime,
    'strategy': str,
    'confidence': float,
    'pnl': float,
    'status': 'open'|'closed'|'cancelled'
}
```

### **Signal Data**

```python
{
    'symbol': str,
    'signal': 'buy'|'sell'|'neutral',
    'confidence': float,         # 0.0-1.0
    'source': str,              # 'technical'|'sentiment'|'earnings'
    'reasoning': str,
    'timestamp': datetime,
    'metadata': dict            # Additional signal-specific data
}
```

---

This API reference provides the foundation for integrating with and extending the trading bot system.
