"""
Hotfixes and workarounds that get applied on startup.
Mostly numpy compat stuff, thread safety wrappers, and config validation.
"""

import numpy as np
import pandas as pd
import threading
import logging
from pathlib import Path

# =============================================================================
# PATCH 1: Fix numpy.NaN issue across all files
# =============================================================================

def apply_numpy_nan_patch():
    """Backfill np.NaN for newer numpy versions."""
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
        print("Applied numpy.NaN patch")

# =============================================================================
# PATCH 2: Thread-safe auto trader management
# =============================================================================

class ThreadSafeAutoTrader:
    """Wraps the auto trader so it can run in a background thread without race conditions."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._instance = None
        self._thread = None
        self._running = False
    
    def start_trader(self, trader_instance):
        with self._lock:
            if self._running:
                return False, "Trader already running"
            
            self._instance = trader_instance
            self._thread = threading.Thread(
                target=self._run_trader,
                daemon=True
            )
            self._thread.start()
            self._running = True
            return True, "Trader started successfully"
    
    def stop_trader(self):
        with self._lock:
            if not self._running:
                return False, "Trader not running"
            
            self._running = False
            if self._instance and hasattr(self._instance, 'stop'):
                self._instance.stop()
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            
            self._instance = None
            self._thread = None
            return True, "Trader stopped successfully"
    
    def is_running(self):
        with self._lock:
            return self._running
    
    def get_status(self):
        with self._lock:
            if not self._running or not self._instance:
                return {"running": False, "trades": []}
            
            return {
                "running": True,
                "trades": getattr(self._instance, 'get_active_trades', lambda: [])(),
                "last_signal": getattr(self._instance, 'get_last_signal', lambda: None)(),
                "pnl": getattr(self._instance, 'get_current_pnl', lambda: 0)()
            }
    
    def _run_trader(self):
        try:
            if self._instance and hasattr(self._instance, 'run'):
                self._instance.run()
        except Exception as e:
            logging.error(f"Error in auto trader thread: {e}")
        finally:
            with self._lock:
                self._running = False

# =============================================================================
# PATCH 3: Database connection safety
# =============================================================================

class SafeDatabaseConnection:
    """SQLite wrapper that locks around queries so threads don't collide."""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def execute_query(self, query, params=None, fetch=False):
        """Run a query with automatic connection open/close."""
        import sqlite3
        
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch:
                    result = cursor.fetchall()
                else:
                    result = cursor.rowcount
                
                conn.commit()
                return result
                
            except Exception as e:
                logging.error(f"Database error: {e}")
                if 'conn' in locals():
                    conn.rollback()
                raise
            finally:
                if 'conn' in locals():
                    conn.close()

# =============================================================================
# PATCH 4: Configuration validation
# =============================================================================

def validate_configuration():
    """Sanity-check the config values before we start trading."""
    errors = []
    warnings = []
    
    try:
        from config import (
            API_KEY, API_SECRET, CAPITAL, RISK_PERCENT,
            MAX_CAPITAL_PER_TRADE, DEFAULT_SYMBOLS
        )
        
        # Required API keys
        if not API_KEY or not API_SECRET:
            errors.append("Missing required Alpaca API credentials")
        
        # Trading parameters
        if CAPITAL <= 0:
            errors.append("CAPITAL must be positive")
        
        if not (0 < RISK_PERCENT <= 10):
            warnings.append("RISK_PERCENT should be between 0-10%")
        
        if not (0 < MAX_CAPITAL_PER_TRADE <= 1):
            errors.append("MAX_CAPITAL_PER_TRADE must be between 0-1")
        
        if not DEFAULT_SYMBOLS:
            warnings.append("No default symbols configured")
        
        # Directory structure
        required_dirs = ['data', 'logs', 'exports']
        for directory in required_dirs:
            Path(directory).mkdir(exist_ok=True)
        
    except ImportError as e:
        errors.append(f"Configuration import error: {e}")
    
    return errors, warnings

# =============================================================================
# PATCH 5: WebSocket connection manager fix
# =============================================================================

class ImprovedWebSocketManager:
    """WebSocket manager with reconnect logic and exponential backoff."""
    
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self._lock = threading.RLock()
        self._connection_attempts = 0
        self._max_attempts = 10
        self._backoff_delay = 5
        self._running = False
        self._ws = None
        
    def start(self):
        """Spin up the connection loop in a daemon thread."""
        with self._lock:
            if self._running:
                return False
            
            self._running = True
            self._connection_attempts = 0
            
            thread = threading.Thread(target=self._connection_loop, daemon=True)
            thread.start()
            return True
    
    def stop(self):
        """Kill the connection."""
        with self._lock:
            self._running = False
            if self._ws:
                try:
                    self._ws.close()
                except:
                    pass
    
    def _connection_loop(self):
        """Keep reconnecting with backoff until max attempts."""
        while self._running and self._connection_attempts < self._max_attempts:
            try:
                self._connect()
                self._connection_attempts = 0  # Reset on successful connection
            except Exception as e:
                logging.error(f"WebSocket connection failed: {e}")
                self._connection_attempts += 1
                
                if self._connection_attempts < self._max_attempts:
                    delay = min(self._backoff_delay * (2 ** self._connection_attempts), 300)
                    logging.info(f"Retrying in {delay} seconds...")
                    threading.Event().wait(delay)
    
    def _connect(self):
        """Actually open the websocket."""
        # Implementation would go here
        pass

# =============================================================================
# Apply all patches
# =============================================================================

def apply_all_patches():
    """Run all the patches in sequence."""
    print("Applying patches...")
    
    apply_numpy_nan_patch()
    
    errors, warnings = validate_configuration()
    
    if errors:
        print("Config errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("Config warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("All patches applied.")
    return True

if __name__ == "__main__":
    apply_all_patches()
