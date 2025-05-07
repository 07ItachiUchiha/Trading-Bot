import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path

class PositionMonitor:
    """
    Monitor positions for exit conditions:
    - Stop loss hits
    - Take profit hits
    - Trailing stops
    - Time-based exits
    - Max loss exits (new)
    """
    def __init__(self):
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._positions = {}  # Current positions to monitor
        self.running = False
    
    def start_monitoring(self):
        """Start the position monitoring thread"""
        if not self.running:
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            self.running = True
    
    def stop_monitoring(self):
        """Stop the position monitoring thread"""
        if self.running:
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
            self.running = False
    
    def add_position(self, position_id, position_data):
        """Add a position to monitor"""
        self._positions[position_id] = position_data
    
    def remove_position(self, position_id):
        """Remove a position from monitoring"""
        if position_id in self._positions:
            del self._positions[position_id]
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._check_all_positions()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Error in position monitor: {e}")
    
    def _check_all_positions(self):
        """Check all positions for exit conditions"""
        positions_to_close = []
        
        for pos_id, pos_data in self._positions.items():
            # Skip already closed positions
            if pos_data.get('status') == 'closed':
                positions_to_close.append(pos_id)
                continue
                
            # Get current price
            symbol = pos_data['symbol']
            current_price = self._get_current_price(symbol)
            
            # Check for stop loss hit
            if pos_data.get('stop_loss'):
                if (pos_data['direction'] == 'long' and current_price <= pos_data['stop_loss']) or \
                   (pos_data['direction'] == 'short' and current_price >= pos_data['stop_loss']):
                    self._close_position(pos_id, current_price, 'stop_loss')
                    continue
            
            # Check for take profit hit
            if pos_data.get('take_profit'):
                if (pos_data['direction'] == 'long' and current_price >= pos_data['take_profit']) or \
                   (pos_data['direction'] == 'short' and current_price <= pos_data['take_profit']):
                    self._close_position(pos_id, current_price, 'take_profit')
                    continue
            
            # Check trailing stop
            if pos_data.get('use_trailing_stop', False):
                trail_pct = pos_data.get('trailing_stop_percentage', 2.0)
                
                # Update highest/lowest seen price
                if pos_data['direction'] == 'long':
                    if 'highest_price' not in pos_data:
                        pos_data['highest_price'] = pos_data['entry_price']
                    
                    if current_price > pos_data['highest_price']:
                        pos_data['highest_price'] = current_price
                        # Update trailing stop level
                        pos_data['trailing_stop'] = current_price * (1 - trail_pct/100)
                        
                    # Check if trailing stop is hit
                    if 'trailing_stop' in pos_data and current_price <= pos_data['trailing_stop']:
                        self._close_position(pos_id, current_price, 'trailing_stop')
                        continue
                else:  # short position
                    if 'lowest_price' not in pos_data:
                        pos_data['lowest_price'] = pos_data['entry_price']
                    
                    if current_price < pos_data['lowest_price']:
                        pos_data['lowest_price'] = current_price
                        # Update trailing stop level
                        pos_data['trailing_stop'] = current_price * (1 + trail_pct/100)
                        
                    # Check if trailing stop is hit
                    if 'trailing_stop' in pos_data and current_price >= pos_data['trailing_stop']:
                        self._close_position(pos_id, current_price, 'trailing_stop')
                        continue
            
            # Check time-based exit
            if pos_data.get('use_time_exit', False) and pos_data.get('exit_time'):
                exit_time = datetime.fromisoformat(pos_data['exit_time'])
                if datetime.now() >= exit_time:
                    self._close_position(pos_id, current_price, 'time_exit')
                    continue
            
            # NEW: Check for max loss exit condition
            if pos_data.get('use_max_loss_exit', False) and pos_data.get('max_loss_percent', 0) > 0:
                entry_price = pos_data.get('entry_price', 0)
                if entry_price > 0:
                    # Calculate current loss percentage
                    if pos_data['direction'] == 'long':
                        loss_percent = (entry_price - current_price) / entry_price * 100
                    else:  # short
                        loss_percent = (current_price - entry_price) / entry_price * 100
                    
                    # If loss exceeds threshold, close the position
                    if loss_percent >= pos_data['max_loss_percent']:
                        self._close_position(pos_id, current_price, 'max_loss_exit')
                        continue
        
        # Clean up closed positions
        for pos_id in positions_to_close:
            self.remove_position(pos_id)
    
    def _close_position(self, position_id, price, reason):
        """Close a position and record the reason"""
        try:
            position = self._positions[position_id]
            symbol = position['symbol']
            direction = position['direction']
            size = position['size']
            
            # Calculate P&L
            if direction == 'long':
                pnl = (price - position['entry_price']) * size
            else:  # short
                pnl = (position['entry_price'] - price) * size
            
            # Record the exit
            position['exit_price'] = price
            position['exit_time'] = datetime.now().isoformat()
            position['exit_reason'] = reason
            position['pnl'] = pnl
            position['status'] = 'closed'
            
            # Execute the exit order
            action = 'sell' if direction == 'long' else 'buy'
            self._execute_exit_order(symbol, action, size, price, position_id, reason)
            
            # Log the exit
            print(f"Closed position {position_id} ({symbol}) due to {reason} - P&L: {pnl:.2f}")
            
        except Exception as e:
            print(f"Error closing position {position_id}: {e}")
    
    def _execute_exit_order(self, symbol, action, size, price, position_id, reason):
        """Execute the exit order"""
        # This would integrate with your trade execution component
        # For now, we'll just log the order
        print(f"EXIT ORDER: {action} {size} {symbol} @ {price} - Reason: {reason}")
        
        # In a real implementation, you would call your order execution function
        # and update position status based on the result
    
    def _get_current_price(self, symbol):
        """Get current price for a symbol"""
        # This would integrate with your market data component
        # For now, return mock prices
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        
        price_map = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
            "ADA": 0.5,
            "DOGE": 0.15,
        }
        
        base_price = price_map.get(base_symbol, 100.0)
        
        # Add small random variation
        variation = np.random.uniform(-0.01, 0.01)  # ±1%
        return base_price * (1 + variation)

def display_position_monitor():
    """Display position monitor controls in the UI"""
    st.subheader("Position Monitor")
    
    # Initialize position monitor in session state if it doesn't exist
    if 'position_monitor' not in st.session_state:
        st.session_state.position_monitor = PositionMonitor()
        
    monitor = st.session_state.position_monitor
    
    # Controls to start/stop monitoring
    col1, col2 = st.columns(2)
    
    with col1:
        if not monitor.running:
            if st.button("Start Monitoring"):
                monitor.start_monitoring()
                st.success("Position monitoring started")
                # Force refresh
                st.rerun()
        else:
            if st.button("Stop Monitoring"):
                monitor.stop_monitoring()
                st.info("Position monitoring stopped")
                # Force refresh
                st.rerun()
    
    with col2:
        status = "Running" if monitor.running else "Stopped"
        status_color = "green" if monitor.running else "red"
        st.markdown(f"Status: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    
    # Display explanatory text
    st.info("""
    Position Monitor automatically manages exits for your open positions:
    
    • **Stop Loss**: Exits when price hits your stop loss level
    • **Take Profit**: Exits when price hits your profit target
    • **Trailing Stop**: Dynamically adjusts stop loss as price moves in your favor
    • **Time Exit**: Automatically exits positions after specified time period
    • **Max Loss Exit**: Automatically closes positions that reach your maximum allowed loss
    
    Start monitoring to enable these automated risk management features.
    """)
