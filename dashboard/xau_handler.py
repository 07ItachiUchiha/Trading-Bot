import pandas as pd
import numpy as np
import logging
import time
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_xau_usd_price():
    """Get current XAU/USD price with fallback to mock data"""
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
        
        # Try alternative API - Alpha Vantage
        try:
            API_KEY = "demo"  # Replace with your Alpha Vantage API key
            url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={API_KEY}"
            response = requests.get(url)
            data = response.json()
            gold_price = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
            logger.info(f"Successfully retrieved XAU/USD price from Alpha Vantage: {gold_price}")
            return gold_price
        except Exception as e:
            logger.warning(f"Failed to fetch XAU/USD from Alpha Vantage: {e}")
        
        # If all APIs fail, use mock data
        return generate_mock_xau_price()
    except Exception as e:
        logger.error(f"Error in get_xau_usd_price: {e}")
        return generate_mock_xau_price()

def generate_mock_xau_price():
    """Generate a realistic mock price for XAU/USD"""
    base_price = 2400.0  # Base gold price in USD
    variation = np.random.normal(0, 10)  # Normal distribution with $10 standard deviation
    mock_price = max(base_price + variation, 1800)  # Ensure price doesn't go too low
    logger.info(f"Generated mock XAU/USD price: {mock_price}")
    return mock_price

def generate_mock_xau_data(timeframe='1h', periods=100):
    """Generate mock XAU/USD historical data"""
    logger.info(f"Generating mock XAU/USD data with {timeframe} timeframe for {periods} periods")
    
    # Current time
    end_date = datetime.now()
    
    # Calculate minutes per bar based on timeframe
    if timeframe == '1m':
        minutes_per_bar = 1
    elif timeframe == '5m':
        minutes_per_bar = 5
    elif timeframe == '15m':
        minutes_per_bar = 15
    elif timeframe == '30m':
        minutes_per_bar = 30
    elif timeframe == '1h':
        minutes_per_bar = 60
    elif timeframe == '4h':
        minutes_per_bar = 240
    elif timeframe == '1d':
        minutes_per_bar = 1440
    else:
        minutes_per_bar = 60  # Default to 1h
    
    # Calculate start date
    start_date = end_date - timedelta(minutes=minutes_per_bar * periods)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Base price with appropriate volatility for gold
    base_price = 2400
    volatility_factor = 0.005  # Gold has relatively low volatility (0.5%)
    
    # Generate price with random walk
    np.random.seed(int(time.time()))  # Randomize seed
    price_changes = np.random.normal(0, base_price * volatility_factor, periods)
    prices = base_price + np.cumsum(price_changes)
    prices = np.maximum(prices, base_price * 0.8)  # Ensure prices don't go too low
    
    # Create DataFrame
    mock_data = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 - 0.002 * np.random.random(periods)),
        'high': prices * (1 + 0.004 * np.random.random(periods)),
        'low': prices * (1 - 0.004 * np.random.random(periods)),
        'close': prices,
        'volume': np.random.randint(1000, 5000, periods)  # Lower volume for gold
    })
    
    # Ensure high is always highest and low is always lowest
    for i in range(len(mock_data)):
        high = max(mock_data.loc[i, 'open'], mock_data.loc[i, 'close'], mock_data.loc[i, 'high'])
        low = min(mock_data.loc[i, 'open'], mock_data.loc[i, 'close'], mock_data.loc[i, 'low'])
        mock_data.loc[i, 'high'] = high
        mock_data.loc[i, 'low'] = low
    
    # Convert datetime to string format
    mock_data['time'] = mock_data['datetime']
    
    logger.info(f"Generated mock XAU/USD data: {len(mock_data)} bars")
    return mock_data
