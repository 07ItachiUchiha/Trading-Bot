import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
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
FINNHUB_WEBHOOK_SECRET = 'd0csqrpr01ql2j3fac90'

# General trading configuration
SYMBOL = 'BTCUSD'  # Default symbol for backward compatibility (Note: Alpaca uses BTCUSD instead of BTCUSDT)
DEFAULT_SYMBOLS = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'ADAUSD', 'SOLUSD', 'DOGEUSD']  # List of tradable symbols
TIMEFRAME = '1H'    # Default timeframe (Note: Alpaca uses uppercase H instead of lowercase h)
MARKET_TYPE = 'crypto' # crypto or stocks

# Capital and risk management
CAPITAL = float(os.getenv('CAPITAL', '10000'))  # Initial trading capital in USD
RISK_PERCENT = float(os.getenv('RISK_PERCENT', '1.0'))  # Risk percentage per trade
MAX_CAPITAL_PER_TRADE = float(os.getenv('MAX_CAPITAL_PER_TRADE', '0.1'))  # Maximum capital allocation per trade (10% of total capital)

# Profit targets
PROFIT_TARGET_PERCENT = float(os.getenv('PROFIT_TARGET_PERCENT', '15.0'))  # Overall profit target percentage
DAILY_PROFIT_TARGET_PERCENT = float(os.getenv('DAILY_PROFIT_TARGET_PERCENT', '3.0'))  # Daily profit target percentage

# Technical analysis parameters
BOLLINGER_LENGTH = 20
BOLLINGER_STD = 2.0
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_SHORT = 50
EMA_LONG = 200
ATR_LENGTH = 14

# Stop loss and take profit settings
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_RR_RATIO = [1.5, 2.5, 4.0]  # Risk:Reward ratios for take profit levels

# News sentiment parameters
NEWS_WEIGHT = 0.5  # Weight for news sentiment signals
SENTIMENT_REFRESH_INTERVAL = 1800  # Refresh sentiment every 30 minutes (in seconds)
NEWS_LOOKBACK_DAYS = 2  # Number of days to look back for news

# Earnings report parameters
EARNINGS_WEIGHT = 0.6  # Weight for earnings report signals
PRE_EVENT_DAYS = 3  # Days before an economic event to consider
POST_EVENT_DAYS = 1  # Days after an economic event to consider

# Notification parameters
SEND_TELEGRAM_ALERTS = True
SEND_DISCORD_ALERTS = False 
SEND_SLACK_ALERTS = False
ALERT_ON_SIGNALS = True
ALERT_ON_TRADES = True

# Watchlist categories
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

# Initialize directory structure
# This can be used to create necessary directories for data storage
EXPORTS_DIR = 'exports'
DATA_DIR = 'data'
LOGS_DIR = 'logs'

# Ensure directories exist
for directory in [EXPORTS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')
LLM_MODEL = os.getenv('LLM_MODEL', 'meta-llama/llama-3-8b-instruct')  # Default model