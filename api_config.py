"""
API Configuration for Trading Bot
Add your API keys here and avoid committing this file to source control
"""

# Market Data APIs
ALPACA_API_KEY = "your_alpaca_key"
ALPACA_API_SECRET = "your_alpaca_secret"

# News and Sentiment APIs
NEWS_API_KEY = "your_news_api_key"
ALPHAVANTAGE_API_KEY = "your_alphavantage_key"
FINNHUB_API_KEY = "your_finnhub_key"

# Additional Recommended APIs
TIINGO_API_KEY = "your_tiingo_key"
POLYGON_API_KEY = "your_polygon_key"
TRADINGVIEW_API_KEY = "your_tradingview_key"
SENTIMENT_TRADER_API_KEY = "your_sentiment_trader_key"

# Create a dictionary for easy access
API_KEYS = {
    'alpaca': {
        'key': ALPACA_API_KEY,
        'secret': ALPACA_API_SECRET
    },
    'newsapi': NEWS_API_KEY,
    'alphavantage': ALPHAVANTAGE_API_KEY,
    'finnhub': FINNHUB_API_KEY,
    'tiingo': TIINGO_API_KEY,
    'polygon': POLYGON_API_KEY,
    'tradingview': TRADINGVIEW_API_KEY,
    'sentiment_trader': SENTIMENT_TRADER_API_KEY
}
