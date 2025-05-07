import json
import logging
import threading
import time
from datetime import datetime
import websocket
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('BinanceWebSocketManager')

class BinanceWebSocketManager:
    """WebSocket client for Binance API"""
    
    def __init__(self):
        """Initialize the Binance WebSocket Manager"""
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.ws = None
        self.websocket_thread = None
        self.running = False
        self.subscriptions = {}
        self.connected = False
        
        # Reconnection parameters
        self.reconnect_attempt = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 5  # Initial delay in seconds
        self.max_delay = 300  # Maximum delay (5 minutes)
        self.last_connect_time = 0
        self.min_time_between_connects = 3  # Minimum time between connection attempts
    
    def start(self):
        """Start the WebSocket connection in a separate thread"""
        if self.running:
            logger.info("BinanceWebSocketManager is already running")
            return
            
        self.running = True
        
        # Start the WebSocket connection in a separate thread
        self.websocket_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.websocket_thread.start()
        
        logger.info("BinanceWebSocketManager started")
    
    def _run_websocket(self):
        """Internal method to establish and maintain the WebSocket connection"""
        while self.running:
            try:
                # Don't attempt to reconnect too rapidly
                current_time = time.time()
                if current_time - self.last_connect_time < self.min_time_between_connects:
                    time.sleep(self.min_time_between_connects)
                
                self.last_connect_time = time.time()
                
                # Build URL with streams if we have subscriptions
                connection_url = self.base_url
                if self.subscriptions:
                    # Get all symbols that have callbacks
                    active_symbols = [symbol for symbol, callbacks in self.subscriptions.items() if callbacks]
                    if active_symbols:
                        streams = [f"{symbol.lower()}@kline_1m" for symbol in active_symbols]
                        connection_url = f"{self.base_url}/stream?streams={'/'.join(streams)}"
                
                # Initialize WebSocket connection
                self.ws = websocket.WebSocketApp(
                    connection_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                # Start WebSocket connection
                logger.info(f"Connecting to Binance WebSocket: {connection_url}")
                self.ws.run_forever()
                
                # If we get here, the connection was closed
                if not self.running:
                    break
                    
                # Calculate backoff delay for reconnection
                if self.reconnect_attempt < self.max_reconnect_attempts:
                    delay = min(self.base_delay * (2 ** self.reconnect_attempt), self.max_delay)
                    logger.info(f"WebSocket disconnected. Reconnecting in {delay} seconds... (Attempt {self.reconnect_attempt + 1}/{self.max_reconnect_attempts})")
                    self.reconnect_attempt += 1
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts. Giving up.")
                    self.running = False
                    break
                
            except Exception as e:
                logger.error(f"Error in BinanceWebSocketManager thread: {e}")
                time.sleep(10)  # Wait before trying again
    
    def _on_open(self, ws):
        """Callback when WebSocket connection is opened"""
        logger.info("Binance WebSocket connected")
        self.connected = True
        self.reconnect_attempt = 0  # Reset reconnection counter
        
        # Subscribe to all symbols
        for symbol, callbacks in self.subscriptions.items():
            if callbacks:
                self._subscribe_to_symbol(symbol)
    
    def _on_message(self, ws, message):
        """Callback when a message is received from the WebSocket"""
        try:
            # Parse message
            data = json.loads(message)
            
            # Handle Binance's message structure
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                data = data['data']
                
                # Parse the stream to get symbol (format: btcusdt@kline_1m)
                parts = stream.split('@')
                if len(parts) >= 1:
                    symbol = parts[0].upper()
                    
                    # Process kline data
                    if 'k' in data and symbol in self.subscriptions:
                        kline = data['k']
                        standardized_data = self._standardize_kline_data(kline, symbol)
                        
                        # Call all registered callbacks for this symbol
                        for callback in self.subscriptions[symbol]:
                            try:
                                callback(standardized_data)
                            except Exception as e:
                                logger.error(f"Error in callback for symbol {symbol}: {e}")
            else:
                # Single stream data format
                if 'e' in data and data['e'] == 'kline':
                    symbol = data['s']
                    kline = data['k']
                    
                    if symbol in self.subscriptions:
                        standardized_data = self._standardize_kline_data(kline, symbol)
                        
                        # Call all registered callbacks for this symbol
                        for callback in self.subscriptions[symbol]:
                            try:
                                callback(standardized_data)
                            except Exception as e:
                                logger.error(f"Error in callback for symbol {symbol}: {e}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _standardize_kline_data(self, kline, symbol):
        """Convert Binance kline data to standardized format"""
        # Extract relevant fields from kline data
        return {
            'symbol': symbol,
            'type': 'bar',
            'time': datetime.fromtimestamp(kline['t'] / 1000).isoformat(),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': kline['x']  # Is this kline closed?
        }
    
    def _on_error(self, ws, error):
        """Callback when an error occurs in the WebSocket"""
        logger.error(f"Binance WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback when WebSocket connection is closed"""
        logger.info(f"Binance WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # If we're shutting down, don't attempt to reconnect
        if not self.running:
            return
    
    def subscribe(self, symbol, callback):
        """Subscribe to updates for a specific symbol"""
        # Format symbol for Binance (e.g., BTCUSDT)
        formatted_symbol = symbol.replace('/', '').upper()
        
        # Add the callback to subscriptions
        if formatted_symbol not in self.subscriptions:
            self.subscriptions[formatted_symbol] = []
        
        # Only add the callback if it's not already subscribed
        if callback not in self.subscriptions[formatted_symbol]:
            self.subscriptions[formatted_symbol].append(callback)
            logger.info(f"Subscribed to Binance: {formatted_symbol}")
        
        # If already connected, subscribe explicitly
        if self.connected:
            self._subscribe_to_symbol(formatted_symbol)
            
        # If not connected, restart connection to include new subscription
        elif self.running and self.ws:
            logger.info(f"Restarting connection to include new subscription: {formatted_symbol}")
            self.ws.close()
    
    def _subscribe_to_symbol(self, symbol):
        """Send subscription message to the WebSocket for a symbol"""
        try:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [
                    f"{symbol.lower()}@kline_1m"  # 1-minute candlestick
                ],
                "id": int(time.time() * 1000)  # Unique ID
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent subscription request for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
    
    def unsubscribe(self, symbol, callback=None):
        """Unsubscribe from updates for a specific symbol"""
        # Format symbol for Binance (e.g., BTCUSDT)
        formatted_symbol = symbol.replace('/', '').upper()
        
        if formatted_symbol in self.subscriptions:
            if callback is None:
                # Remove all callbacks for this symbol
                self.subscriptions[formatted_symbol] = []
                if self.connected:
                    self._unsubscribe_from_symbol(formatted_symbol)
                logger.info(f"Unsubscribed from all callbacks for: {formatted_symbol}")
            else:
                # Remove specific callback
                if callback in self.subscriptions[formatted_symbol]:
                    self.subscriptions[formatted_symbol].remove(callback)
                    logger.info(f"Unsubscribed from: {formatted_symbol}")
                
                # If no callbacks left, unsubscribe from the symbol
                if not self.subscriptions[formatted_symbol] and self.connected:
                    self._unsubscribe_from_symbol(formatted_symbol)
    
    def _unsubscribe_from_symbol(self, symbol):
        """Send unsubscribe message to the WebSocket for a symbol"""
        try:
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": [
                    f"{symbol.lower()}@kline_1m"  # 1-minute candlestick
                ],
                "id": int(time.time() * 1000)  # Unique ID
            }
            self.ws.send(json.dumps(unsubscribe_msg))
            logger.info(f"Sent unsubscription request for {symbol}")
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
    
    def stop(self):
        """Stop the WebSocket connection"""
        logger.info("Stopping BinanceWebSocketManager")
        self.running = False
        
        # Close WebSocket connection
        if self.ws:
            self.ws.close()
        
        # Clear subscriptions
        self.subscriptions = {}
        
        logger.info("BinanceWebSocketManager stopped")

    def add_to_dataframe(self, df, new_data):
        """Add a new data point to an existing DataFrame"""
        # If the data is for an already closed candle, append it
        if new_data['is_closed']:
            new_row = pd.DataFrame({
                'time': [pd.to_datetime(new_data['time'])],
                'open': [new_data['open']],
                'high': [new_data['high']],
                'low': [new_data['low']],
                'close': [new_data['close']],
                'volume': [new_data['volume']]
            })
            return pd.concat([df, new_row], ignore_index=True)
        
        # If the data is for a current candle, update the last row if timestamps match
        # or append if it's a new timestamp
        if not df.empty:
            last_time = df.iloc[-1]['time']
            current_time = pd.to_datetime(new_data['time'])
            
            # If the timestamps match, update the last row
            if pd.to_datetime(last_time) == current_time:
                df.iloc[-1, df.columns.get_indexer(['high'])[0]] = max(df.iloc[-1]['high'], new_data['high'])
                df.iloc[-1, df.columns.get_indexer(['low'])[0]] = min(df.iloc[-1]['low'], new_data['low'])
                df.iloc[-1, df.columns.get_indexer(['close'])[0]] = new_data['close']
                df.iloc[-1, df.columns.get_indexer(['volume'])[0]] = new_data['volume']
                return df
                
        # Otherwise, add as new row
        new_row = pd.DataFrame({
            'time': [pd.to_datetime(new_data['time'])],
            'open': [new_data['open']],
            'high': [new_data['high']],
            'low': [new_data['low']],
            'close': [new_data['close']],
            'volume': [new_data['volume']]
        })
        return pd.concat([df, new_row], ignore_index=True)
