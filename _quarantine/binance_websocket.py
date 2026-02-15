import json
import logging
import threading
import time
from datetime import datetime
import websocket
import pandas as pd

logger = logging.getLogger('BinanceWebSocketManager')

class BinanceWebSocketManager:
    """WS client for Binance live price streams."""
    
    def __init__(self):
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.ws = None
        self.websocket_thread = None
        self.running = False
        self.subscriptions = {}
        self.connected = False
        
        # reconnection
        self.reconnect_attempt = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 5
        self.max_delay = 300
        self.last_connect_time = 0
        self.min_time_between_connects = 3
    
    def start(self):
        """Spin up the WS connection thread."""
        if self.running:
            logger.info("BinanceWebSocketManager is already running")
            return
            
        self.running = True
        
        # Start in a background thread
        self.websocket_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.websocket_thread.start()
        
        logger.info("BinanceWebSocketManager started")
    
    def _run_websocket(self):
        """Main loop that keeps the connection alive."""
        while self.running:
            try:
                # Don't reconnect too fast
                current_time = time.time()
                if current_time - self.last_connect_time < self.min_time_between_connects:
                    time.sleep(self.min_time_between_connects)
                
                self.last_connect_time = time.time()
                
                connection_url = self.base_url
                if self.subscriptions:
                    active_symbols = [symbol for symbol, callbacks in self.subscriptions.items() if callbacks]
                    if active_symbols:
                        streams = [f"{symbol.lower()}@kline_1m" for symbol in active_symbols]
                        connection_url = f"{self.base_url}/stream?streams={'/'.join(streams)}"
                
                # Set up the WS app
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
                    
                # Calculate backoff delay
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
        """WS connected successfully."""
        logger.info("Binance WebSocket connected")
        self.connected = True
        self.reconnect_attempt = 0
        
        # resubscribe
        for symbol, callbacks in self.subscriptions.items():
            if callbacks:
                self._subscribe_to_symbol(symbol)
    
    def _on_message(self, ws, message):
        """Handle incoming WS messages."""
        try:
            data = json.loads(message)
            
            # Binance wraps data in stream/data when multi-streaming
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                data = data['data']
                
                # Parse stream name to get the symbol (format: btcusdt@kline_1m)
                parts = stream.split('@')
                if len(parts) >= 1:
                    symbol = parts[0].upper()
                    
                    # Process kline data
                    if 'k' in data and symbol in self.subscriptions:
                        kline = data['k']
                        standardized_data = self._standardize_kline_data(kline, symbol)
                        
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
        """Convert Binance kline fields to our standard format."""
        # Extract relevant fields
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
        """Log WS errors."""
        logger.error(f"Binance WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WS close."""
        logger.info(f"Binance WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # If we're shutting down, don't attempt to reconnect
        if not self.running:
            return
    
    def subscribe(self, symbol, callback):
        """Subscribe to live updates for a symbol."""
        formatted_symbol = symbol.replace('/', '').upper()
        
        # Add callback
        if formatted_symbol not in self.subscriptions:
            self.subscriptions[formatted_symbol] = []
        
        # Only add if not already there
        if callback not in self.subscriptions[formatted_symbol]:
            self.subscriptions[formatted_symbol].append(callback)
            logger.info(f"Subscribed to Binance: {formatted_symbol}")
        
        # If already connected, subscribe now
        if self.connected:
            self._subscribe_to_symbol(formatted_symbol)
            
        # Otherwise reconnect to pick up the new sub
        elif self.running and self.ws:
            logger.info(f"Restarting connection to include new subscription: {formatted_symbol}")
            self.ws.close()
    
    def _subscribe_to_symbol(self, symbol):
        """Send a sub message over the WS."""
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
        """Remove a subscription for a symbol."""
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
        """Send an unsub message over the WS."""
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
        """Shut everything down."""
        logger.info("Stopping BinanceWebSocketManager")
        self.running = False
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Clear subscriptions
        self.subscriptions = {}
        
        logger.info("BinanceWebSocketManager stopped")

    def add_to_dataframe(self, df, new_data):
        """Append or update a bar in the dataframe."""
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
