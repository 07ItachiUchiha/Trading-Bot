"""
Utility functions to standardize symbol handling across the trading bot
"""

def standardize_symbol(symbol, destination='alpaca'):
    """
    Standardize symbol formatting based on destination system
    
    Args:
        symbol (str): The symbol to format
        destination (str): The destination system ('alpaca', 'binance', etc)
    
    Returns:
        str: Properly formatted symbol for the destination
    """
    
    # Remove any whitespace
    symbol = symbol.strip()
    
    if destination.lower() == 'alpaca':
        # Alpaca uses BTC/USD format for crypto
        if symbol.endswith('USD') and '/' not in symbol:
            # BTCUSD -> BTC/USD
            return f"{symbol[:-3]}/USD"
        elif symbol.endswith('USDT') and '/' not in symbol:
            # BTCUSDT -> BTC/USDT
            return f"{symbol[:-4]}/USDT"
        return symbol
        
    elif destination.lower() == 'binance':
        # Binance uses BTCUSDT format (no slashes)
        if '/' in symbol:
            return symbol.replace('/', '')
        return symbol
        
    elif destination.lower() == 'display':
        # For display to users, use the slash format
        if symbol.endswith('USD') and '/' not in symbol:
            return f"{symbol[:-3]}/USD"
        elif symbol.endswith('USDT') and '/' not in symbol:
            return f"{symbol[:-4]}/USDT"
        return symbol
    
    # Default case - return as is
    return symbol
    
def get_base_currency(symbol):
    """Extract the base currency from a symbol"""
    if '/' in symbol:
        return symbol.split('/')[0]
    elif symbol.endswith('USD'):
        return symbol[:-3]
    elif symbol.endswith('USDT'):
        return symbol[:-4]
    return symbol

def get_quote_currency(symbol):
    """Extract the quote currency from a symbol"""
    if '/' in symbol:
        return symbol.split('/')[1]
    elif symbol.endswith('USD'):
        return 'USD'
    elif symbol.endswith('USDT'):
        return 'USDT'
    return 'USD'  # Default quote currency

def is_crypto(symbol):
    """Determine if a symbol represents a cryptocurrency"""
    base = get_base_currency(symbol)
    common_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOGE', 'XRP', 'DOT']
    
    return base in common_cryptos or 'USD' in symbol or 'USDT' in symbol
    
def get_initial_price(symbol):
    """Get an approximate initial price for a symbol"""
    base = get_base_currency(symbol)
    
    price_map = {
        'BTC': 30000,
        'ETH': 2000,
        'BNB': 500,
        'ADA': 0.6,
        'SOL': 140,
        'DOGE': 0.15,
        'XRP': 0.5,
        'DOT': 10,
        'AAPL': 180,
        'MSFT': 350,
        'GOOGL': 130,
        'AMZN': 140,
    }
    
    return price_map.get(base, 100)  # Default to 100 if unknown
