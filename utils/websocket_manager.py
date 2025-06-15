import websocket
import threading
import json
import time
import logging
import random
from datetime import datetime
import pandas as pd
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('WebSocketManager')

class WebSocketManager:
    """
    WebSocket Manager for handling real-time data connections
    
    This class manages a WebSocket connection, with automatic reconnection,
    exponential backoff, and subscription management.
    """
    
    # Class-level variables for singleton pattern
    _instances = {}
    _lock = threading.RLock()
    _connection_times = deque(maxlen=10)  # Track recent connection attempts
    
    @classmethod
    def get_instance(cls, api_key, api_secret, base_url="wss://stream.data.alpaca.markets/v2"):
        """Get singleton instance - use this instead of constructor"""
        # Use a key based on the API key and base URL to allow different instances for different accounts
        key = f"{api_key[-4:]}:{base_url}"
        
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = WebSocketManager(api_key, api_secret, base_url)
            return cls._instances[key]
    
    @classmethod
    def _track_connection(cls):
        """Track connection attempt for rate limiting"""
        now = time.time()
        cls._connection_times.append(now)
    
    @classmethod
    def _should_rate_limit(cls):
        """Check if we should rate limit connections"""
        if len(cls._connection_times) < 2:
            return False
            
        # If we have 5+ connections in 10 seconds, slow down
        if len(cls._connection_times) >= 5:
            time_span = time.time() - cls._connection_times[0]
            if time_span < 10:
                return True
        
        return False
    
    def __init__(self, api_key, api_secret, base_url="wss://stream.data.alpaca.markets/v2"):
        """Initialize the WebSocket manager with credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws = None
        self.websocket_thread = None
        self.running = False
        self.subscriptions = {}
        self.connected = False
        self._connection_attempt = 0  # Track connection attempts for this instance
        
        # Reconnection parameters
        self.reconnect_attempt = 0
        self.max_reconnect_attempts = 15
        self.base_delay = 5  # Initial delay in seconds
        self.max_delay = 300  # Maximum delay (5 minutes)
        self.last_connect_time = 0
        self.min_time_between_connects = 10  # Minimum time between connection attempts
        
        # Heartbeat tracking
        self.last_heartbeat = None
        self.heartbeat_interval = 30  # Seconds between heartbeats
        self.heartbeat_missed = 0
        self.max_missed_heartbeats = 3
        
        # Message queue for when connection is lost
        self.message_queue = []
        self.max_queue_size = 100
        
        # Connection state tracking
        self._connecting = False
        self._current_connection_id = 0
        self._should_reconnect = True
        self.last_error = None
        
        # Network quality and failure tracking for adaptive reconnection
        self.network_quality = "good"  # Assume good quality initially
        self.consecutive_failures = 0  # Track consecutive failure counts
        self.connection_history = []  # Track history of connection attempts
        
        # Network condition tracking
        self.min_backoff_delay = 1    # Minimum backoff delay in seconds
        self.max_backoff_delay = 300  # Maximum backoff delay (5 minutes)
        self.adaptive_factor = 1.0    # Will be adjusted based on network quality

    def start(self):
        """Start the WebSocket connection in a separate thread"""
        with self._lock:
            if self.running:
                logger.info("WebSocketManager is already running")
                return
                
            self.running = True
            self._should_reconnect = True
            
            # Start the WebSocket connection in a separate thread
            if self.websocket_thread is None or not self.websocket_thread.is_alive():
                self.websocket_thread = threading.Thread(target=self._run_websocket, daemon=True)
                self.websocket_thread.start()
                
                # Start the heartbeat monitor thread
                self.heartbeat_thread = threading.Thread(target=self._monitor_heartbeat, daemon=True)
                self.heartbeat_thread.start()
                logger.info("WebSocketManager started")
            else:
                logger.info("WebSocketManager thread already running")
    
    def _run_websocket(self):
        """Internal method to establish and maintain the WebSocket connection"""
        while self.running:
            try:
                # Check if we're connecting too frequently
                if WebSocketManager._should_rate_limit():
                    delay = random.uniform(15, 30)  # Random delay between 15-30 seconds
                    logger.warning(f"Rate limiting detected - delaying connection for {delay:.1f} seconds")
                    time.sleep(delay)
                
                # Don't attempt to reconnect too rapidly
                current_time = time.time()
                time_since_last = current_time - self.last_connect_time
                if time_since_last < self.min_time_between_connects:
                    sleep_time = self.min_time_between_connects - time_since_last
                    logger.info(f"Waiting {sleep_time:.1f} seconds before next connection attempt")
                    time.sleep(sleep_time)
                
                # Prevent multiple connection attempts simultaneously
                if self._connecting:
                    logger.info("Connection attempt already in progress, waiting...")
                    time.sleep(5)
                    continue
                
                # Track this connection
                with self._lock:
                    WebSocketManager._track_connection()
                    self.last_connect_time = time.time()
                    self._connecting = True
                    self._current_connection_id += 1
                    current_connection = self._current_connection_id
                    self._connection_attempt += 1
                
                # Initialize WebSocket connection
                logger.info(f"Connecting to WebSocket... (Attempt: {self._connection_attempt}, ID: {current_connection})")
                websocket.enableTrace(False)  # Disable trace to reduce log noise
                
                # Set timeout for connection - increase timeout to prevent rapid reconnections
                websocket.setdefaulttimeout(30)
                
                # Initialize connection with appropriate headers and user agent
                headers = [
                    "User-Agent: TradingBot/1.0",
                    "Accept-Encoding: gzip, deflate, br",
                    "Cache-Control: no-cache"
                ]
                
                self.ws = websocket.WebSocketApp(
                    self.base_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    header=headers
                )
                
                # Start WebSocket connection with increased ping interval and timeout
                self.ws.run_forever(ping_interval=15, ping_timeout=10)
                
                # Connection has closed
                with self._lock:
                    self._connecting = False
                
                # If we're shutting down or current connection is obsolete, break
                if not self.running or not self._should_reconnect or current_connection < self._current_connection_id:
                    logger.info("WebSocket manager stopping or superseded by newer connection")
                    break
                
                # Calculate backoff delay for reconnection - using much longer delays when hitting rate limits
                if "429" in str(self.last_error) or "too many" in str(self.last_error).lower():
                    # Rate limit hit - use a much longer delay (1-3 minutes)
                    delay = random.uniform(60, 180)
                    logger.warning(f"Rate limit hit. Using extended delay: {delay:.1f} seconds")
                    time.sleep(delay)
                    self.reconnect_attempt += 1
                elif self.reconnect_attempt < self.max_reconnect_attempts:
                    # Exponential backoff with jitter
                    max_backoff = min(self.base_delay * (2 ** self.reconnect_attempt), self.max_delay)
                    delay = max_backoff * (0.5 + random.random())
                    logger.info(f"WebSocket disconnected. Reconnecting in {delay:.1f} seconds... (Attempt {self.reconnect_attempt + 1}/{self.max_reconnect_attempts})")
                    self.reconnect_attempt += 1
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts. Giving up.")
                    self.running = False
                    break
                
                # Log connection attempt info
                self.connection_history.append({
                    'attempt': self._connection_attempt,
                    'success': not self.running,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(self.last_error) if not self.running else None
                })
            except Exception as e:
                logger.error(f"Error in WebSocketManager thread: {e}")
                with self._lock:
                    self._connecting = False
                    self.last_error = str(e)
                time.sleep(10)  # Wait before trying again
        
    def _monitor_heartbeat(self):
        """Monitor heartbeats to ensure the connection is healthy"""
        while self.running:
            try:
                if self.connected and self.last_heartbeat:
                    elapsed = time.time() - self.last_heartbeat
                    
                    # Adaptive heartbeat interval based on network quality
                    adaptive_interval = self.heartbeat_interval
                    if self.network_quality == "poor":
                        adaptive_interval = self.heartbeat_interval * 1.5
                    
                    if elapsed > adaptive_interval:
                        self.heartbeat_missed += 1
                        logger.warning(f"Missed heartbeat ({self.heartbeat_missed}/{self.max_missed_heartbeats})")
                        
                        # Add more detailed diagnostics
                        logger.info(f"Time since last heartbeat: {elapsed:.1f}s, "
                                   f"Network quality: {self.network_quality}, "
                                   f"Consecutive failures: {self.consecutive_failures}")
                        
                        # Send ping to check connection
                        if self.ws and self.ws.sock:
                            try:
                                self.ws.sock.ping()
                                logger.info("Sent ping to WebSocket server")
                            except:
                                logger.error("Failed to send ping")
                        
                        # If too many heartbeats missed, force reconnection
                        if self.heartbeat_missed >= self.max_missed_heartbeats:
                            # Record consecutive failures for adaptive delay calculation
                            self.consecutive_failures += 1
                            logger.warning(f"Too many missed heartbeats, reconnecting... "
                                          f"(Consecutive failures: {self.consecutive_failures})")
                            self._force_reconnect()
                    else:
                        # Reset consecutive failures on successful heartbeats
                        if self.consecutive_failures > 0 and elapsed < adaptive_interval/2:
                            self.consecutive_failures = max(0, self.consecutive_failures - 1)
            
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                time.sleep(5)

    def _force_reconnect(self):
        """Force a reconnection by closing the current socket"""
        with self._lock:
            try:
                self._current_connection_id += 1  # Invalidate current connection
                if self.ws:
                    self.ws.close()
                    logger.info("Forced WebSocket reconnection")
            
                # Add network quality assessment - NEW
                self.network_quality = self._assess_network_quality()
                logger.info(f"Current network quality assessment: {self.network_quality}")
            except Exception as e:
                logger.error(f"Error forcing reconnection: {e}")
            finally:
                self.connected = False
                self.heartbeat_missed = 0
                self._connecting = False

    def _assess_network_quality(self):
        """Assess network quality based on recent connection history"""
        # NEW METHOD to assess network quality
        if len(self._connection_times) < 2:
            return "unknown"
        
        # Calculate time between disconnects
        time_diffs = []
        for i in range(1, len(self._connection_times)):
            time_diffs.append(self._connection_times[i] - self._connection_times[i-1])
    
        if not time_diffs:
            return "unknown"
        
        avg_time_between_disconnects = sum(time_diffs) / len(time_diffs)
        
        # Classify network quality
        if avg_time_between_disconnects > 300:  # More than 5 minutes between disconnects
            return "good"
        elif avg_time_between_disconnects > 60:  # More than 1 minute
            return "fair"
        else:
            return "poor"

    def _on_open(self, ws):
        """Callback when WebSocket connection is opened"""
        logger.info(f"WebSocket connected (ID: {self._current_connection_id})")
        with self._lock:
            self.connected = True
            self.reconnect_attempt = 0  # Reset reconnection counter
            self.heartbeat_missed = 0  # Reset missed heartbeat counter
            self.last_heartbeat = time.time()
            
            # Authenticate
            auth_data = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            ws.send(json.dumps(auth_data))
            
            # Wait a moment before subscribing to avoid overwhelming the server
            time.sleep(0.5)
            
            # Re-subscribe to all symbols with rate limiting
            symbols = list(self.subscriptions.keys())
            for symbol in symbols:
                if self.subscriptions[symbol]:
                    self._subscribe_to_symbol(symbol)
                    time.sleep(0.1)  # Small delay between subscriptions

    def _on_message(self, ws, message):  # ws param is required by websocket-client library
        """Callback when a message is received from the WebSocket"""
        try:
            # Update heartbeat time
            self.last_heartbeat = time.time()
            self.heartbeat_missed = 0
            
            # Parse message
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                for item in data:
                    self._process_message(item)
            else:
                self._process_message(data)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def _process_message(self, data):
        """Process a single message/update from the WebSocket"""
        try:
            msg_type = data.get('T', data.get('type', 'unknown'))
            
            # Handle authentication response
            if msg_type == 'auth':
                if data.get('msg') == 'authenticated':
                    logger.info("WebSocket authenticated successfully")
                else:
                    logger.error(f"Authentication failed: {data}")
            
            # Handle error messages
            elif msg_type == 'error':
                logger.error(f"WebSocket error message: {data}")
                
                # Check for rate limit errors
                error_msg = str(data.get('msg', '')).lower()
                if '429' in error_msg or 'too many' in error_msg or 'rate limit' in error_msg:
                    logger.warning("Rate limit detected, increasing backoff")
                    self.min_time_between_connects = min(self.min_time_between_connects * 2, 60)
            
            # Handle data updates
            elif msg_type in ['trade', 'quote', 'bar']:
                symbol = data.get('S', data.get('symbol'))
                if symbol and symbol in self.subscriptions:
                    # Convert the message to a standardized format for callbacks
                    standardized_data = self._standardize_data(data, msg_type)
                    
                    # Call all registered callbacks for this symbol
                    for callback in self.subscriptions[symbol]:
                        try:
                            callback(standardized_data)
                        except Exception as e:
                            logger.error(f"Error in callback for symbol {symbol}: {e}")
            
            # Handle heartbeat messages
            elif msg_type == 'heartbeat' or msg_type == 'status':
                pass  # Just update the last_heartbeat time
            
            # Handle unknown message types
            else:
                logger.debug(f"Unhandled message type: {msg_type}, data: {data}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _standardize_data(self, data, msg_type):
        """Standardize different data formats to a common structure"""
        try:
            result = {
                'raw': data,  # Keep the raw data for reference
                'type': msg_type,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Extract symbol
            symbol = data.get('S', data.get('symbol', None))
            result['symbol'] = symbol
            
            # Process based on message type
            if msg_type == 'trade':
                result['price'] = float(data.get('p', data.get('price', 0)))
                result['size'] = float(data.get('s', data.get('size', 0)))
                result['time'] = data.get('t', data.get('timestamp', datetime.now().isoformat()))
                result['exchange'] = data.get('x', data.get('exchange', 'unknown'))
                result['trade_id'] = data.get('i', data.get('id', 0))
                
            elif msg_type == 'quote':
                result['bid_price'] = float(data.get('bp', data.get('bid_price', 0)))
                result['bid_size'] = float(data.get('bs', data.get('bid_size', 0)))
                result['ask_price'] = float(data.get('ap', data.get('ask_price', 0)))
                result['ask_size'] = float(data.get('as', data.get('ask_size', 0)))
                result['time'] = data.get('t', data.get('timestamp', datetime.now().isoformat()))
                
            elif msg_type == 'bar':
                result['open'] = float(data.get('o', data.get('open', 0)))
                result['high'] = float(data.get('h', data.get('high', 0)))
                result['low'] = float(data.get('l', data.get('low', 0)))
                result['close'] = float(data.get('c', data.get('close', 0)))
                result['volume'] = float(data.get('v', data.get('volume', 0)))
                result['time'] = data.get('t', data.get('timestamp', datetime.now().isoformat()))
                result['is_closed'] = data.get('ic', data.get('is_closed', True))
                
            # Support XAU/USD specifically - make sure gold prices are handled properly
            if symbol and ('XAU' in symbol or 'GOLD' in symbol):
                if 'price' in result and result['price'] == 0:
                    from dashboard.xau_handler import get_xau_usd_price
                    result['price'] = get_xau_usd_price()
                    logger.info(f"Replaced zero XAU price with mock data: {result['price']}")
                    
            return result
        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            # Return original data if standardization fails
            return {
                'raw': data,
                'type': msg_type,
                'error': str(e)
            }

    def _on_error(self, ws, error):  # ws param is required by websocket-client library
        """Callback when an error occurs in the WebSocket"""
        logger.error(f"WebSocket error: {error}")
        
        # Store the error for reference
        self.last_error = error
        
        # Handle specific error types
        error_str = str(error).lower()
        if "429" in error_str or "too many requests" in error_str:
            # Rate limit hit - increase backoff significantly
            self.min_time_between_connects = min(self.min_time_between_connects * 2, 120)
            logger.warning(f"Rate limit hit, increasing connection delay to {self.min_time_between_connects}s")
            
            # Sleep thread to prevent immediate reconnection attempts
            time.sleep(30)

    def _on_close(self, ws, close_status_code, close_msg):  # ws param is required by websocket-client library
        """Callback when WebSocket connection is closed"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # If we're shutting down, don't attempt to reconnect
        if not self.running or not self._should_reconnect:
            return

    def subscribe(self, symbol, callback):
        """Subscribe to updates for a specific symbol"""
        with self._lock:
            # Add the callback to subscriptions
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = []
            
            # Only add the callback if it's not already subscribed
            if callback not in self.subscriptions[symbol]:
                self.subscriptions[symbol].append(callback)
                logger.info(f"Subscribed to: {symbol}")
            
            # If already connected, send the subscribe message
            if self.connected and self.ws:
                self._subscribe_to_symbol(symbol)

    def _subscribe_to_symbol(self, symbol):
        """Send subscription message to the WebSocket"""
        try:
            # For crypto
            if '/' in symbol:
                sub_message = {
                    "action": "subscribe",
                    "trades": [symbol],
                    "quotes": [symbol],
                    "bars": [symbol]
                }
            # For stocks
            else:
                sub_message = {
                    "action": "subscribe",
                    "trades": [symbol],
                    "quotes": [symbol],
                    "bars": [symbol]
                }
        
            self.ws.send(json.dumps(sub_message))
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")

    def unsubscribe(self, symbol, callback=None):
        """Unsubscribe from updates for a specific symbol"""
        if symbol in self.subscriptions:
            if callback is None:
                # Remove all callbacks for this symbol
                self.subscriptions[symbol] = []
                if self.connected:
                    self._unsubscribe_from_symbol(symbol)
                logger.info(f"Unsubscribed from all callbacks for: {symbol}")
            else:
                # Remove specific callback
                if callback in self.subscriptions[symbol]:
                    self.subscriptions[symbol].remove(callback)
                    logger.info(f"Unsubscribed from: {symbol}")
                
                # If no callbacks left, unsubscribe from the symbol
                if not self.subscriptions[symbol] and self.connected:
                    self._unsubscribe_from_symbol(symbol)

    def _unsubscribe_from_symbol(self, symbol):
        """Send unsubscribe message to the WebSocket"""
        try:
            # For crypto
            if '/' in symbol:
                unsub_message = {
                    "action": "unsubscribe",
                    "trades": [symbol],
                    "quotes": [symbol],
                    "bars": [symbol]
                }
            # For stocks
            else:
                unsub_message = {
                    "action": "unsubscribe",
                    "trades": [symbol],
                    "quotes": [symbol],
                    "bars": [symbol]
                }
        
            self.ws.send(json.dumps(unsub_message))
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")

    def stop(self):
        """Stop the WebSocket connection"""
        logger.info("Stopping WebSocketManager")
        self.running = False
    
        # Close WebSocket connection
        if self.ws:
            self.ws.close()
    
        # Clear subscriptions
        self.subscriptions = {}
    
        # Wait for thread to terminate if necessary
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=5.0)
            
        logger.info("WebSocketManager stopped")

    def add_to_dataframe(self, df, new_data):
        """Add a new data point to an existing DataFrame"""
        try:
            # If the data is for a closed candle, append it
            is_closed = new_data.get('is_closed', True)  # Default to True if not specified
            
            # Convert timestamp to pandas datetime
            try:
                new_time = pd.to_datetime(new_data.get('time', datetime.now().isoformat()))
            except:
                new_time = datetime.now()
            
            # Create a new row with the received data
            new_row = pd.DataFrame({
                'time': [new_time],
                'open': [float(new_data.get('open', 0))],
                'high': [float(new_data.get('high', 0))],
                'low': [float(new_data.get('low', 0))],
                'close': [float(new_data.get('close', 0))],
                'volume': [float(new_data.get('volume', 0))]
            })
            
            # If the dataframe is empty or we have a new closed candle, append it
            if df.empty or is_closed:
                return pd.concat([df, new_row], ignore_index=True)
            
            # If this is for an existing candle (not closed), update the last row
            # First, check if we have the same timestamp
            if not df.empty:
                last_time = pd.to_datetime(df.iloc[-1]['time'])
                
                # If the timestamps match, update the last row
                if last_time == new_time:
                    df.loc[df.index[-1], 'high'] = max(df.iloc[-1]['high'], new_row.iloc[0]['high'])
                    df.loc[df.index[-1], 'low'] = min(df.iloc[-1]['low'], new_row.iloc[0]['low'])
                    df.loc[df.index[-1], 'close'] = new_row.iloc[0]['close']
                    df.loc[df.index[-1], 'volume'] = new_row.iloc[0]['volume']
                    return df
                else:
                    # New timestamp but not a closed candle, append it
                    return pd.concat([df, new_row], ignore_index=True)
                
            return df
        except Exception as e:
            logger.error(f"Error adding data to dataframe: {e}")
            return df  # Return original dataframe on error

    def _get_backoff_multiplier(self):
        """Get adaptive backoff multiplier based on network quality and history"""
        if self.network_quality == "good":
            return 0.5  # Faster reconnects for good networks
        elif self.network_quality == "poor":
            return 2.0  # Slower reconnects for poor networks
    
        # Adjust based on consecutive failures
        if self.consecutive_failures > 5:
            return 1.5
        elif self.consecutive_failures > 10:
            return 2.5
    
        return 1.0

# Example usage
if __name__ == "__main__":
    # os is not directly used but often imported in main for environment setup
    import os  # noqa
    from config import API_KEY, API_SECRET
    
    # Define a sample callback
    def handle_update(data):
        print(f"Received update: {data}")
    
    # Create and start the WebSocketManager
    ws_manager = WebSocketManager(API_KEY, API_SECRET)
    ws_manager.start()
    
    # Subscribe to BTC/USD updates
    ws_manager.subscribe("BTC/USD", handle_update)
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop WebSocketManager when Ctrl+C is pressed
        ws_manager.stop()
        print("WebSocketManager stopped")