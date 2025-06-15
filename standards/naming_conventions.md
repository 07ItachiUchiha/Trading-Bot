# Trading Bot Naming Conventions

## 📋 **Standardized Naming Conventions**

### **1. Function Naming**
- Use `snake_case` for all function and method names
- Use descriptive names that indicate purpose
- Prefix with verb to indicate action

**Examples:**
```python
# ✅ Good
def fetch_historical_data()
def calculate_position_size()
def analyze_market_sentiment()
def update_trade_status()

# ❌ Avoid
def fetchHistoricalData()  # camelCase
def calc_pos()           # abbreviated
def getData()           # too generic
```

### **2. Class Naming**
- Use `PascalCase` for class names
- Use descriptive names ending with type/purpose

**Examples:**
```python
# ✅ Good
class AutoTradingManager
class SentimentAnalyzer
class RiskManager
class WebSocketManager

# ❌ Avoid  
class autoTradingManager  # camelCase
class TradingBot         # too generic
class WSManager          # abbreviated
```

### **3. Variable Naming**
- Use `snake_case` for variables
- Use descriptive names
- Constants use `UPPER_CASE`

**Examples:**
```python
# ✅ Good
trade_signal = "BUY"
confidence_score = 0.85
API_KEY = "your_key"
MAX_POSITION_SIZE = 0.1

# ❌ Avoid
sig = "BUY"              # abbreviated
TradeSignal = "BUY"      # PascalCase for variable
apiKey = "your_key"      # camelCase
```

### **4. File Naming**
- Use `snake_case` for Python files
- Group related functionality

**Examples:**
```
# ✅ Good
sentiment_analyzer.py
risk_management.py
market_data_manager.py
trading_strategy.py

# ❌ Avoid
SentimentAnalyzer.py     # PascalCase
riskMgmt.py             # abbreviated
tradingbot.py           # too generic
```

### **5. Module/Package Naming**
- Use lowercase with underscores
- Keep short but descriptive

**Examples:**
```
strategy/
utils/
dashboard/
backtest/
```

## 🔧 **Implementation Guidelines**

1. **Refactor existing inconsistent names gradually**
2. **Use IDE refactoring tools when possible**
3. **Update all references when renaming**
4. **Document naming changes in version control**
