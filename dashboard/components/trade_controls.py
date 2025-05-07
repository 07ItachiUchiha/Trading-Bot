import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import API configuration
try:
    from config import API_KEY, API_SECRET
except ImportError:
    API_KEY = os.environ.get("ALPACA_API_KEY", "")
    API_SECRET = os.environ.get("ALPACA_API_SECRET", "")

# Global variables for trade execution tracking
pending_orders = {}
order_updates = {}
trade_lock = threading.RLock()

def execute_trade(symbol, action, quantity, price=None, order_type="market"):
    """
    Execute a buy or sell trade
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        action (str): 'buy' or 'sell'
        quantity (float): Trade quantity
        price (float, optional): Limit price (for limit orders)
        order_type (str): Type of order ('market', 'limit')
    
    Returns:
        dict: Trade result with status and details
    """
    # Validate inputs
    if not symbol or not action or not quantity:
        return {
            "status": "error",
            "message": "Missing required parameters"
        }
    
    if action.lower() not in ['buy', 'sell']:
        return {
            "status": "error",
            "message": "Action must be 'buy' or 'sell'"
        }
    
    # Generate unique order ID
    order_id = str(uuid.uuid4())
    
    try:
        # Check if we have enough balance for this trade
        # In a real implementation, this would check actual account balances
        
        # For demo purposes, we'll simulate trades with a slight delay
        trade_thread = threading.Thread(
            target=_process_trade_simulation,
            args=(order_id, symbol, action, quantity, price, order_type)
        )
        trade_thread.daemon = True
        trade_thread.start()
        
        # Return immediately with pending status
        return {
            "status": "pending",
            "order_id": order_id,
            "message": f"{action.capitalize()} order submitted",
            "symbol": symbol,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Trade execution failed: {str(e)}"
        }

def _process_trade_simulation(order_id, symbol, action, quantity, price, order_type):
    """Simulate trade processing with a realistic delay"""
    global pending_orders, order_updates
    
    with trade_lock:
        # Record the pending order
        pending_orders[order_id] = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    
    # Simulate network delay and processing time (0.5-2 seconds)
    time.sleep(np.random.uniform(0.5, 2.0))
    
    # Determine execution price (with some slippage for realism)
    if not price:
        # For market orders, determine a realistic price
        base_price = _get_current_price(symbol)
        slippage = base_price * np.random.uniform(-0.001, 0.002)  # 0.1% to 0.2% slippage
        executed_price = base_price + slippage if action.lower() == "buy" else base_price - slippage
    else:
        # For limit orders, use the specified price
        executed_price = price
    
    # Calculate profit/loss if this is a closing trade
    pnl = 0
    if _is_closing_trade(symbol, action, quantity):
        pnl = _calculate_pnl(symbol, action, quantity, executed_price)
    
    # Record the completed order
    with trade_lock:
        order_result = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "requested_price": price,
            "executed_price": executed_price,
            "order_type": order_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl
        }
        
        # Update the order status
        pending_orders[order_id]["status"] = "completed"
        pending_orders[order_id]["executed_price"] = executed_price
        pending_orders[order_id]["pnl"] = pnl
        
        # Add to order updates
        order_updates[order_id] = order_result
        
        # Save the trade to history
        _save_trade_to_history(order_result)

def _get_current_price(symbol):
    """Get the current price for a symbol (simulated for demo)"""
    # In a real implementation, this would fetch the current price from an API
    base_prices = {
        "BTC/USD": 95000,
        "ETH/USD": 3500,
        "SOL/USD": 150,
        "AAPL": 190,
        "MSFT": 420,
        "GOOGL": 180,
        "AMZN": 200
    }
    
    # Use a predefined price or generate a random one
    base_price = base_prices.get(symbol, 100)
    
    # Add some randomness to simulate market movement
    return base_price * (1 + np.random.uniform(-0.005, 0.005))

def _is_closing_trade(symbol, action, quantity):
    """Check if this trade is closing an existing position (simulated)"""
    # In a real implementation, this would check actual positions
    return np.random.choice([True, False], p=[0.3, 0.7])  # 30% chance it's a closing trade

def _calculate_pnl(symbol, action, quantity, price):
    """Calculate profit/loss for a closing trade (simulated)"""
    # In a real implementation, this would calculate the actual P&L
    # For demo, generate a random P&L between -5% and +5% of position value
    position_value = price * quantity
    return position_value * np.random.uniform(-0.05, 0.05)

def _save_trade_to_history(trade_data):
    """Save trade details to history file"""
    exports_dir = Path(__file__).parent.parent.parent / "exports"
    
    # Create exports directory if it doesn't exist
    exports_dir.mkdir(exist_ok=True)
    
    # Create or update the trade history file
    history_file = exports_dir / "trade_history.json"
    
    try:
        # Load existing history if available
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {"trades": []}
        
        # Add the new trade
        history["trades"].append(trade_data)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Error saving trade to history: {e}")

def get_pending_orders():
    """Get the current pending orders"""
    with trade_lock:
        return dict(pending_orders)

def get_order_updates():
    """Get recently completed orders and clear the updates"""
    with trade_lock:
        updates = dict(order_updates)
        order_updates.clear()
        return updates

def get_position_summary():
    """Get a summary of open positions (simulated for demo)"""
    # In a real implementation, this would fetch actual positions from the API
    
    # Generate some realistic position data
    symbols = ["BTC/USD", "ETH/USD", "AAPL", "MSFT"]
    positions = []
    
    for symbol in symbols:
        # Randomly decide if we have a position in this symbol
        if np.random.random() < 0.5:
            # Determine position details
            direction = np.random.choice(["long", "short"])
            entry_price = _get_current_price(symbol) * (1 + np.random.uniform(-0.1, 0.1))
            current_price = _get_current_price(symbol)
            size = round(np.random.uniform(0.1, 2.0), 2)
            
            # Calculate P&L
            if direction == "long":
                unrealized_pnl = (current_price - entry_price) * size
            else:
                unrealized_pnl = (entry_price - current_price) * size
                
            # Create position record
            positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "size": size,
                "unrealized_pnl": unrealized_pnl,
                "pnl_percentage": (unrealized_pnl / (entry_price * size)) * 100
            })
    
    return positions

def display_trade_controls(symbol=None, default_quantity=None):
    """Display trade control panel with buy and sell buttons"""
    st.subheader("Trade Controls")
    
    # Quick wallet balance check
    try:
        from dashboard.components.wallet import load_account_data
        account = load_account_data()
        
        # Extract base currency from symbol
        if symbol:
            if '/' in symbol:
                base_currency = symbol.split('/')[0]
                quote_currency = symbol.split('/')[1]
            else:
                # Handle BTCUSD format
                if symbol.endswith('USD'):
                    base_currency = symbol[:-3]
                    quote_currency = 'USD'
                else:
                    base_currency = symbol
                    quote_currency = 'USD'
                    
            # Show relevant balances
            col1, col2 = st.columns(2)
            with col1:
                base_balance = account['balance'].get(base_currency, 0)
                st.metric(f"{base_currency} Balance", f"{base_balance:.6f}")
            with col2:
                quote_balance = account['balance'].get(quote_currency, 0)
                st.metric(f"{quote_currency} Balance", f"${quote_balance:.2f}")
    except Exception as e:
        # Skip balance display if error
        pass
    
    # Trade Input Form
    with st.form(key="trade_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Symbol selection
            if not symbol:
                symbol = st.text_input("Symbol", value="BTC/USD", 
                                      help="Enter trading symbol (e.g., BTC/USD, AAPL)")
            else:
                st.text_input("Symbol", value=symbol, disabled=True)
        
        with col2:
            # Quantity input
            quantity = st.number_input(
                "Quantity",
                min_value=0.0001, 
                max_value=1000.0,
                value=default_quantity if default_quantity else 0.01,
                step=0.01,
                format="%.4f",
                help="Enter the quantity to trade"
            )
        
        # Order type selection
        col1, col2 = st.columns(2)
        
        with col1:
            order_type = st.selectbox(
                "Order Type",
                options=["Market", "Limit"],
                index=0,
                help="Market: execute immediately at current price. Limit: execute only at specified price or better."
            )
        
        with col2:
            # Price input (only for limit orders)
            price = None
            if order_type == "Limit":
                price = st.number_input(
                    "Limit Price",
                    min_value=0.01,
                    value=_get_current_price(symbol),
                    step=0.01,
                    format="%.2f",
                    help="Enter the limit price"
                )
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            buy_submitted = st.form_submit_button(
                "BUY", 
                use_container_width=True,
                type="primary",
                help="Execute a buy order"
            )
        
        with col2:
            sell_submitted = st.form_submit_button(
                "SELL",
                use_container_width=True,
                help="Execute a sell order"
            )
    
    # Handle form submission
    if buy_submitted or sell_submitted:
        action = "buy" if buy_submitted else "sell"
        
        # Execute the trade
        result = execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price if order_type == "Limit" else None,
            order_type=order_type.lower()
        )
        
        # Show the result
        if result["status"] == "pending":
            st.success(f"{action.upper()} order submitted! Order ID: {result['order_id']}")
        else:
            st.error(f"Trade execution failed: {result['message']}")
    
    # Display open positions
    display_positions()
    
    # Display recent orders
    display_orders()

def display_positions():
    """Display open positions"""
    positions = get_position_summary()
    
    st.subheader("Open Positions")
    
    if not positions:
        st.info("No open positions")
        return
    
    # Create a DataFrame for display
    df = pd.DataFrame(positions)
    
    # Format the dataframe columns
    if not df.empty:
        df['entry_price'] = df['entry_price'].map('${:,.2f}'.format)
        df['current_price'] = df['current_price'].map('${:,.2f}'.format)
        df['unrealized_pnl'] = df['unrealized_pnl'].map('${:,.2f}'.format)
        df['pnl_percentage'] = df['pnl_percentage'].map('{:,.2f}%'.format)
    
    # Display as a table
    st.dataframe(
        df,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "direction": st.column_config.TextColumn("Direction"),
            "entry_price": st.column_config.TextColumn("Entry Price"),
            "current_price": st.column_config.TextColumn("Current Price"),
            "size": st.column_config.NumberColumn("Size", format="%.4f"),
            "unrealized_pnl": st.column_config.TextColumn("Unrealized P&L"),
            "pnl_percentage": st.column_config.TextColumn("P&L %"),
        },
        hide_index=True,
        use_container_width=True
    )

def display_orders():
    """Display recent and pending orders"""
    # Get pending and recent orders
    pending = get_pending_orders()
    updates = get_order_updates()
    
    st.subheader("Recent Orders")
    
    if not pending and not updates:
        st.info("No recent orders")
        return
    
    # Combine pending and completed orders
    all_orders = list(pending.values()) + list(updates.values())
    
    # Convert to DataFrame
    if all_orders:
        df = pd.DataFrame(all_orders)
        
        # Sort by timestamp (newest first)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format prices and amounts
        for col in ['executed_price', 'requested_price', 'pnl']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
        
        # Display as table
        st.dataframe(
            df,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol"),
                "action": st.column_config.TextColumn("Action"),
                "status": st.column_config.TextColumn("Status"),
                "quantity": st.column_config.NumberColumn("Quantity", format="%.4f"),
                "executed_price": st.column_config.TextColumn("Executed Price"),
                "requested_price": st.column_config.TextColumn("Requested Price"),
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "pnl": st.column_config.TextColumn("P&L"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No recent orders")
