import pandas as pd
import numpy as np
import logging
import time
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_xau_usd_price():
    """Get current XAU/USD price from available APIs"""
    try:
        # Try to fetch from multiple sources
        # First try Forex-Python, a reliable API for forex data
        try:
            from forex_python.converter import CurrencyRates
            c = CurrencyRates()
            # XAU is gold price in troy ounce, we need to convert
            gold_price = c.get_rate('XAU', 'USD')
            logger.info(f"Successfully retrieved XAU/USD price: {gold_price}")
            return gold_price
        except Exception as e:
            logger.warning(f"Failed to fetch XAU/USD from forex_python: {e}")
        
        # Try alternative API - Alpha Vantage (requires configured API key)
        try:
            from config import ALPHAVANTAGE_API_KEY
            if not ALPHAVANTAGE_API_KEY:
                raise RuntimeError("ALPHAVANTAGE_API_KEY not configured in .env")
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={ALPHAVANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            gold_price = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
            logger.info(f"Successfully retrieved XAU/USD price from Alpha Vantage: {gold_price}")
            return gold_price
        except Exception as e:
            logger.warning(f"Failed to fetch XAU/USD from Alpha Vantage: {e}")
        
        # If all APIs fail, raise error
        raise RuntimeError("Unable to fetch XAU/USD price from any available API. Please check API keys and network connection.")
    except Exception as e:
        logger.error(f"Error retrieving XAU/USD price: {e}")
        raise
    return mock_data
