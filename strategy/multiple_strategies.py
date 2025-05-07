import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger('multiple_strategies')

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        
    def generate_signals(self, df):
        """
        Generate trading signals based on the strategy
        
        Args:
            df (pd.DataFrame): Price data with indicators
            
        Returns:
            dict: Dictionary with signals and indicators
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_visual_explanation(self):
        """
        Get a visual explanation of the strategy
        
        Returns:
            str: Markdown explanation of how the strategy works
        """
        return self.description

class RSI_EMA_Strategy(TradingStrategy):
    """RSI + EMA Crossover Strategy"""
    
    def __init__(self):
        description = """
        ## RSI + EMA Crossover Strategy
        
        This strategy combines two popular technical indicators:
        
        1. **RSI (Relative Strength Index)** - A momentum oscillator that measures the speed and change of price movements
        2. **EMA (Exponential Moving Average)** - A type of moving average that places greater weight on recent data
        
        ### Buy Signal Conditions:
        - RSI crosses above 30 (oversold condition)
        - Price crosses above EMA(50)
        
        ### Sell Signal Conditions:
        - RSI crosses above 70 (overbought condition)
        - Price crosses below EMA(50)
        
        The combination provides both trend direction (EMA) and momentum (RSI) confirmation.
        """
        super().__init__("RSI + EMA Crossover", description)
    
    def generate_signals(self, df):
        """Generate signals based on RSI and EMA crossover"""
        signals = {}
        data = df.copy()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate EMAs
        data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
        
        # Store indicators in signals
        signals['rsi'] = data['rsi']
        signals['ema50'] = data['ema50']
        
        # Generate buy signals
        buy_signals = pd.Series(False, index=data.index)
        sell_signals = pd.Series(False, index=data.index)
        
        for i in range(1, len(data)):
            # Buy when RSI crosses above 30 and price crosses above EMA50
            if (data['rsi'].iloc[i-1] < 30 and data['rsi'].iloc[i] > 30) and \
               (data['close'].iloc[i-1] < data['ema50'].iloc[i-1] and data['close'].iloc[i] > data['ema50'].iloc[i]):
                buy_signals.iloc[i] = True
            
            # Sell when RSI crosses above 70 and price crosses below EMA50
            if (data['rsi'].iloc[i-1] < 70 and data['rsi'].iloc[i] > 70) and \
               (data['close'].iloc[i-1] > data['ema50'].iloc[i-1] and data['close'].iloc[i] < data['ema50'].iloc[i]):
                sell_signals.iloc[i] = True
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        
        # Generate combined signal based on most recent data
        combined_signal = {
            'signal': 'neutral',
            'confidence': 0.0,
            'reasoning': 'No clear signal'
        }
        
        # Check for recent buy/sell signals (last 3 candles)
        recent_buy = buy_signals.iloc[-3:].any()
        recent_sell = sell_signals.iloc[-3:].any()
        
        if recent_buy and not recent_sell:
            combined_signal = {
                'signal': 'buy',
                'confidence': 0.7,
                'reasoning': 'RSI crossed above 30 and price crossed above EMA(50)'
            }
        elif recent_sell and not recent_buy:
            combined_signal = {
                'signal': 'sell',
                'confidence': 0.7,
                'reasoning': 'RSI crossed above 70 and price crossed below EMA(50)'
            }
        
        signals['combined_signal'] = combined_signal
        return signals

class BollingerRSI_Strategy(TradingStrategy):
    """Bollinger Bands + RSI Strategy"""
    
    def __init__(self):
        description = """
        ## Bollinger Bands + RSI Strategy
        
        This strategy combines two popular technical indicators:
        
        1. **Bollinger Bands** - Dynamic volatility bands that expand and contract based on price action
        2. **RSI (Relative Strength Index)** - A momentum oscillator that measures the speed and change of price movements
        
        ### Buy Signal Conditions:
        - Price touches or crosses below the lower Bollinger Band
        - RSI is below 30 (oversold condition)
        
        ### Sell Signal Conditions:
        - Price touches or crosses above the upper Bollinger Band
        - RSI is above 70 (overbought condition)
        
        This strategy works well in ranging markets as it identifies potential reversals at support and resistance levels.
        """
        super().__init__("Bollinger Bands + RSI", description)
    
    def generate_signals(self, df):
        """Generate signals based on Bollinger Bands and RSI"""
        signals = {}
        data = df.copy()
        
        # Calculate Bollinger Bands (20-period, 2 standard deviations)
        window = 20
        data['sma20'] = data['close'].rolling(window=window).mean()
        data['std20'] = data['close'].rolling(window=window).std()
        data['upper_band'] = data['sma20'] + (data['std20'] * 2)
        data['middle_band'] = data['sma20']
        data['lower_band'] = data['sma20'] - (data['std20'] * 2)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Store indicators in signals
        signals['rsi'] = data['rsi']
        signals['upper_band'] = data['upper_band']
        signals['middle_band'] = data['middle_band']
        signals['lower_band'] = data['lower_band']
        
        # Generate buy signals
        buy_signals = pd.Series(False, index=data.index)
        sell_signals = pd.Series(False, index=data.index)
        
        for i in range(1, len(data)):
            # Buy when price closes below lower band and RSI is below 30
            if (data['close'].iloc[i] < data['lower_band'].iloc[i]) and (data['rsi'].iloc[i] < 30):
                buy_signals.iloc[i] = True
            
            # Sell when price closes above upper band and RSI is above 70
            if (data['close'].iloc[i] > data['upper_band'].iloc[i]) and (data['rsi'].iloc[i] > 70):
                sell_signals.iloc[i] = True
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        
        # Generate combined signal based on most recent data
        combined_signal = {
            'signal': 'neutral',
            'confidence': 0.0,
            'reasoning': 'No clear signal'
        }
        
        # Check for recent buy/sell signals (last 3 candles)
        recent_buy = buy_signals.iloc[-3:].any()
        recent_sell = sell_signals.iloc[-3:].any()
        
        if recent_buy and not recent_sell:
            combined_signal = {
                'signal': 'buy',
                'confidence': 0.7,
                'reasoning': 'Price below lower Bollinger Band with RSI oversold'
            }
        elif recent_sell and not recent_buy:
            combined_signal = {
                'signal': 'sell',
                'confidence': 0.7,
                'reasoning': 'Price above upper Bollinger Band with RSI overbought'
            }
        
        signals['combined_signal'] = combined_signal
        return signals

class BreakoutDetectionStrategy(TradingStrategy):
    """Breakout Detection Strategy"""
    
    def __init__(self):
        description = """
        ## Breakout Detection Strategy
        
        This strategy detects breakouts from periods of price consolidation:
        
        1. **Consolidation Period** - Identified by narrowing Bollinger Band width (low volatility)
        2. **Breakout** - Detected when volatility expands rapidly after consolidation
        
        ### Buy Signal Conditions:
        - Price breaks out upward from a consolidation period
        - Volume increases significantly
        
        ### Sell Signal Conditions:
        - Price breaks out downward from a consolidation period
        - Volume increases significantly
        
        This strategy works well in trending markets as it catches the beginning of new trends after periods of consolidation.
        """
        super().__init__("Breakout Detection", description)
    
    def generate_signals(self, df):
        """Generate signals based on breakout detection"""
        from strategy.strategy import detect_consolidation, detect_breakout
        
        signals = {}
        data = df.copy()
        
        # Detect consolidation periods
        consolidation, bb_width, atr = detect_consolidation(data, lookback_period=20, threshold=0.03)
        signals['bb_width'] = bb_width
        
        # Detect breakouts
        buy_signals, sell_signals = detect_breakout(data, consolidation, bb_width, lookback=15)
        
        # Calculate Bollinger Bands for visualization
        window = 20
        data['sma20'] = data['close'].rolling(window=window).mean()
        data['std20'] = data['close'].rolling(window=window).std()
        data['upper_band'] = data['sma20'] + (data['std20'] * 2)
        data['middle_band'] = data['sma20'] 
        data['lower_band'] = data['sma20'] - (data['std20'] * 2)
        
        # Store indicators in signals
        signals['upper_band'] = data['upper_band']
        signals['middle_band'] = data['middle_band']
        signals['lower_band'] = data['lower_band']
        signals['atr'] = atr
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        
        # Generate combined signal based on most recent data
        combined_signal = {
            'signal': 'neutral',
            'confidence': 0.0,
            'reasoning': 'No clear signal'
        }
        
        # Check for recent buy/sell signals (last 5 candles)
        recent_buy = buy_signals.iloc[-5:].any()
        recent_sell = sell_signals.iloc[-5:].any()
        
        if recent_buy and not recent_sell:
            combined_signal = {
                'signal': 'buy',
                'confidence': 0.75,
                'reasoning': 'Upward breakout from consolidation period detected'
            }
        elif recent_sell and not recent_buy:
            combined_signal = {
                'signal': 'sell',
                'confidence': 0.75,
                'reasoning': 'Downward breakout from consolidation period detected'
            }
        
        signals['combined_signal'] = combined_signal
        return signals

# Dictionary of all available strategies
AVAILABLE_STRATEGIES = {
    "rsi_ema": RSI_EMA_Strategy(),
    "bollinger_rsi": BollingerRSI_Strategy(),
    "breakout": BreakoutDetectionStrategy()
}

def get_strategy_by_name(name):
    """Get a strategy instance by name"""
    return AVAILABLE_STRATEGIES.get(name)

def get_available_strategy_names():
    """Get a list of available strategy names"""
    return list(AVAILABLE_STRATEGIES.keys())

def get_available_strategy_display_names():
    """Get a list of available strategy display names"""
    return [strategy.name for strategy in AVAILABLE_STRATEGIES.values()]
