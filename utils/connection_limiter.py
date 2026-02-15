import time
from collections import deque
import threading
import logging

logger = logging.getLogger('ConnectionLimiter')

class ConnectionLimiter:
    """Singleton rate limiter to avoid 429 errors from APIs."""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ConnectionLimiter()
            return cls._instance
    
    def __init__(self):
        self.connection_times = deque(maxlen=100)
        self.connection_windows = {
            '1s': {'limit': 2, 'window': 1},       # Max 2 connections per second
            '10s': {'limit': 5, 'window': 10},     # Max 5 connections per 10 seconds
            '60s': {'limit': 15, 'window': 60},    # Max 15 connections per minute
            '5m': {'limit': 30, 'window': 300},    # Max 30 connections per 5 minutes
        }
    
    def track_connection(self):
        """Log a connection attempt."""
        with self._lock:
            self.connection_times.append(time.time())
    
    def should_limit(self):
        """True if we've been connecting too fast."""
        with self._lock:
            now = time.time()
            
            # Check each time window
            for window_name, config in self.connection_windows.items():
                limit = config['limit']
                window = config['window']
                
                count = sum(1 for t in self.connection_times if now - t < window)
                
                if count >= limit:
                    logger.warning(f"Rate limit hit for {window_name} window: {count} connections (limit: {limit})")
                    return True
            
            return False
    
    def get_suggested_wait_time(self):
        """How long to wait before the next connection attempt."""
        with self._lock:
            if not self.connection_times:
                return 0
                
            # default 1s wait
            wait_time = 1
            now = time.time()
            
            # Check each time window
            for window_name, config in self.connection_windows.items():
                window = config['window']
                
                # Get times in this window
                times_in_window = [t for t in self.connection_times if now - t < window]
                count = len(times_in_window)
                
                if count >= config['limit'] and times_in_window:
                    # when will the oldest connection expire from this window
                    oldest = min(times_in_window)
                    time_until_free = (oldest + window) - now
                    wait_time = max(wait_time, time_until_free + 1)  # Add 1s margin
            
            return wait_time
