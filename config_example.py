# Example configuration file
# Copy this file to config.py and add your API keys

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API keys
API_KEY = os.getenv('ALPACA_API_KEY', '')  # Alpaca API key
API_SECRET = os.getenv('ALPACA_API_SECRET', '')  # Alpaca API secret
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')  # NewsAPI key
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')  # AlphaVantage API key
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')  # Finnhub API key
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

# Webhook secrets
FINNHUB_WEBHOOK_SECRET = os.getenv('FINNHUB_WEBHOOK_SECRET', 'your_webhook_secret')

# General trading configuration
SYMBOL = 'BTCUSD'  # Default symbol for backward compatibility
DEFAULT_SYMBOLS = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'ADAUSD', 'SOLUSD', 'DOGEUSD']  # List of tradable symbols
TIMEFRAME = '1H'    # Default timeframe (Note: Alpaca uses uppercase H instead of lowercase h)
MARKET_TYPE = 'crypto'  # crypto or stocks

# Capital and risk management
CAPITAL = float(os.getenv('CAPITAL', '10000'))  # Initial trading capital in USD
RISK_PERCENT = float(os.getenv('RISK_PERCENT', '1.0'))  # Risk percentage per trade
MAX_CAPITAL_PER_TRADE = float(os.getenv('MAX_CAPITAL_PER_TRADE', '0.1'))  # Maximum capital allocation per trade (10% of total capital)

# Profit targets
PROFIT_TARGET_PERCENT = float(os.getenv('PROFIT_TARGET_PERCENT', '15.0'))  # Overall profit target percentage
DAILY_PROFIT_TARGET_PERCENT = float(os.getenv('DAILY_PROFIT_TARGET_PERCENT', '3.0'))  # Daily profit target percentage

# Technical analysis parameters
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30'))
EMA_FAST = int(os.getenv('EMA_FAST', '9'))
EMA_SLOW = int(os.getenv('EMA_SLOW', '21'))
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD = float(os.getenv('BB_STD', '2.0'))
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP', '2.0'))  # Trailing stop percentage

# Strategy parameters
TECHNICAL_WEIGHT = 0.7  # Weight for technical signals
NEWS_WEIGHT = 0.4  # Weight for news-based signals
EARNINGS_WEIGHT = 0.6  # Weight for earnings signals

# Sentiment analysis parameters
SENTIMENT_REFRESH_INTERVAL = 60  # Minutes between sentiment updates
NEWS_LOOKBACK_DAYS = 3  # Days to look back for news
MIN_NEWS_COUNT = 3  # Minimum number of news items to consider

# Initialize directory structure
EXPORTS_DIR = 'exports'
DATA_DIR = 'data'
LOGS_DIR = 'logs'

# Create directories if they don't exist
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
