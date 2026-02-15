import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# API keys
API_KEY = os.getenv('ALPACA_API_KEY', '')
API_SECRET = os.getenv('ALPACA_API_SECRET', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

# Webhook secrets
FINNHUB_WEBHOOK_SECRET = os.getenv('FINNHUB_WEBHOOK_SECRET', '')

# Trading defaults
SYMBOL = 'BTCUSD'  # Alpaca uses BTCUSD not BTCUSDT
DEFAULT_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XAU/USD']
TIMEFRAME = '1H'    # uppercase H for Alpaca
MARKET_TYPE = 'crypto'

# Capital and risk
CAPITAL = float(os.getenv('CAPITAL', '10000'))
RISK_PERCENT = float(os.getenv('RISK_PERCENT', '1.0'))
MAX_CAPITAL_PER_TRADE = float(os.getenv('MAX_CAPITAL_PER_TRADE', '0.1'))

# Profit targets
PROFIT_TARGET_PERCENT = float(os.getenv('PROFIT_TARGET_PERCENT', '15.0'))
DAILY_PROFIT_TARGET_PERCENT = float(os.getenv('DAILY_PROFIT_TARGET_PERCENT', '3.0'))

# TA params
BOLLINGER_LENGTH = 20
BOLLINGER_STD = 2.0
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_SHORT = 12
EMA_LONG = 200
ATR_LENGTH = 14

# Stop loss / take profit
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_RR_RATIO = [1.5, 2.5, 4.0]

# News / sentiment
NEWS_WEIGHT = 0.5
SENTIMENT_REFRESH_INTERVAL = 1800  # 30 min
NEWS_LOOKBACK_DAYS = 2

# Earnings
EARNINGS_WEIGHT = 0.6
PRE_EVENT_DAYS = 3
POST_EVENT_DAYS = 1

# Notifications
SEND_TELEGRAM_ALERTS = True
SEND_DISCORD_ALERTS = False 
SEND_SLACK_ALERTS = False
ALERT_ON_SIGNALS = True
ALERT_ON_TRADES = True

# Watchlist
WATCHLIST_CATEGORIES = {
    'crypto_major': 'Major Cryptocurrencies',
    'crypto_alts': 'Alternative Cryptocurrencies',
    'crypto_defi': 'DeFi Tokens',
    'stocks_tech': 'Technology Stocks',
    'stocks_finance': 'Financial Stocks',
    'stocks_consumer': 'Consumer Stocks',
    'indices': 'Market Indices',
    'forex': 'Forex Pairs',
    'commodities': 'Commodities',
}

# Directory setup
EXPORTS_DIR = 'exports'
DATA_DIR = 'data'
LOGS_DIR = 'logs'

# Ensure dirs exist
for directory in [EXPORTS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'meta-llama/llama-4-maverick:free')  # Default model
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
