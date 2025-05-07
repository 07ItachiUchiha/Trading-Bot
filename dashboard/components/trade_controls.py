import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
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

def execute_trade(symbol, action, quantity, price=None, order_type="market", risk_settings=None):
    """
    Execute a buy or sell trade with risk management
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        action (str): 'buy' or 'sell'
        quantity (float): Trade quantity
        price (float, optional): Limit price (for limit orders)
        order_type (str): Type of order ('market', 'limit')
        risk_settings (dict): Risk management settings
    
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
        # For demo purposes, we'll simulate trades with a slight delay
        trade_thread = threading.Thread(
            target=_process_trade_simulation,
            args=(order_id, symbol, action, quantity, price, order_type, risk_settings)
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

def _is_closing_trade(symbol, action, quantity):
    """Check if this trade closes an existing position"""
    try:
        account_file = Path(__file__).parent.parent.parent / "data" / "account.json"
        
        if not account_file.exists():
            return False
            
        with open(account_file, 'r') as f:
            account = json.load(f)
        
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol.replace('USD', '')
        
        # Check if we have a position in this asset
        if base_symbol in account['balance']:
            current_balance = account['balance'][base_symbol]
            
            # For sell actions, we're closing if we have a balance
            if action.lower() == 'sell' and current_balance > 0:
                return True
                
        return False
    except Exception as e:
        print(f"Error in _is_closing_trade: {e}")
        return False

def _calculate_pnl(symbol, action, quantity, executed_price):
    """Calculate profit/loss for a closing trade"""
    try:
        account_file = Path(__file__).parent.parent.parent / "data" / "account.json"
        
        if not account_file.exists():
            return 0.0
            
        with open(account_file, 'r') as f:
            account = json.load(f)
        
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol.replace('USD', '')
        
        # Only calculate PnL for sell actions
        if action.lower() != 'sell':
            return 0.0
            
        # Get entry transactions to calculate average price
        entry_transactions = []
        for transaction in account.get('transactions', []):
            if transaction.get('symbol') == base_symbol and transaction.get('type') == 'buy':
                entry_transactions.append(transaction)
                
        if not entry_transactions:
            return 0.0
            
        # Calculate average entry price
        total_cost = sum(t.get('amount', 0) * t.get('price', 0) for t in entry_transactions)
        total_amount = sum(t.get('amount', 0) for t in entry_transactions)
        
        if total_amount == 0:
            return 0.0
            
        avg_entry_price = total_cost / total_amount
        
        # Calculate PnL
        pnl = (executed_price - avg_entry_price) * min(quantity, total_amount)
        return pnl
        
    except Exception as e:
        print(f"Error calculating PnL: {e}")
        return 0.0

def _process_trade_simulation(order_id, symbol, action, quantity, price, order_type, risk_settings=None):
    """Simulate trade processing with risk management"""
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
    
    # Record the completed order with risk management details
    with trade_lock:
        order_result = {
            "id": order_id,
            "symbol": symbol,
            "action": action,
            "size": quantity,
            "price": executed_price,
            "requested_price": price,
            "order_type": order_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl
        }
        
        # Add risk management details if provided
        if risk_settings:
            order_result["stop_loss"] = risk_settings.get("stop_loss")
            order_result["take_profit"] = risk_settings.get("take_profit")
            order_result["trailing_stop"] = risk_settings.get("use_trailing_stop")
            order_result["time_exit"] = risk_settings.get("use_time_exit")
            
            # NEW: Add max loss exit settings
            order_result["use_max_loss_exit"] = risk_settings.get("use_max_loss_exit")
            order_result["max_loss_percent"] = risk_settings.get("max_loss_percent")
            
            if risk_settings.get("use_time_exit"):
                exit_time = datetime.now() + timedelta(hours=risk_settings.get("exit_hours", 24))
                order_result["exit_time"] = exit_time.isoformat()
        
        # Update the order status
        pending_orders[order_id]["status"] = "completed"
        pending_orders[order_id]["executed_price"] = executed_price
        pending_orders[order_id]["pnl"] = pnl
        
        # Add to order updates
        order_updates[order_id] = order_result
        
        # Save the trade to history
        _save_trade_to_history(order_result)

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
            
        # Update account balance
        _update_account_balance(trade_data)
            
    except Exception as e:
        print(f"Error saving trade to history: {e}")

def _update_account_balance(trade_data):
    """Update account balance after a trade"""
    try:
        account_file = Path(__file__).parent.parent.parent / "data" / "account.json"
        
        # Create data directory if it doesn't exist
        account_file.parent.mkdir(exist_ok=True)
        
        # Load current account data
        if account_file.exists():
            with open(account_file, 'r') as f:
                account = json.load(f)
        else:
            # Initialize with default values
            account = {
                "balance": {
                    "USD": 10000.0,
                    "BTC": 0.0,
                    "ETH": 0.0,
                    "SOL": 0.0,
                    "ADA": 0.0,
                    "DOGE": 0.0,
                    "BNB": 0.0
                },
                "equity": 10000.0,
                "margin_used": 0.0,
                "transactions": []
            }
        
        # Process trade impact on balance
        symbol = trade_data["symbol"].split("/")[0] if "/" in trade_data["symbol"] else trade_data["symbol"]
        
        # Record transaction
        transaction = {
            "id": trade_data["id"],
            "type": trade_data["action"],
            "symbol": symbol,
            "amount": trade_data["size"],
            "price": trade_data["price"],
            "timestamp": trade_data["timestamp"],
            "status": "completed"
        }
        
        # Update balances based on trade type
        if trade_data["action"] == "buy":
            # Deduct USD for the purchase
            cost = trade_data["price"] * trade_data["size"]
            account["balance"]["USD"] -= cost
            
            # Add the cryptocurrency
            if symbol in account["balance"]:
                account["balance"][symbol] += trade_data["size"]
            else:
                account["balance"][symbol] = trade_data["size"]
                
            transaction["usd_value"] = -cost
                
        elif trade_data["action"] == "sell":
            # Add USD from the sale
            proceeds = trade_data["price"] * trade_data["size"]
            account["balance"]["USD"] += proceeds
            
            # Subtract the cryptocurrency
            if symbol in account["balance"]:
                account["balance"][symbol] = max(0, account["balance"][symbol] - trade_data["size"])
                
            transaction["usd_value"] = proceeds
            
        # Add PnL if this is a position close
        if "pnl" in trade_data:
            transaction["pnl"] = trade_data["pnl"]
        
        # Add transaction to history
        if "transactions" not in account:
            account["transactions"] = []
            
        account["transactions"].append(transaction)
        
        # Recalculate equity
        equity = account["balance"]["USD"]
        for crypto, amount in account["balance"].items():
            if crypto != "USD" and amount > 0:
                # Get latest price if available, otherwise use the trade price
                if crypto == symbol:
                    price = trade_data["price"]
                else:
                    # Use cached price or a placeholder
                    price = _get_current_price(f"{crypto}/USD")
                
                equity += amount * price
        
        account["equity"] = equity
        
        # Save updated account data
        with open(account_file, 'w') as f:
            json.dump(account, f, indent=2)
            
    except Exception as e:
        print(f"Error updating account balance: {e}")

def get_position_summary():
    """Get a summary of open positions with real PnL calculations"""
    try:
        account_file = Path(__file__).parent.parent.parent / "data" / "account.json"
        
        if not account_file.exists():
            return []
            
        # Load account data
        with open(account_file, 'r') as f:
            account = json.load(f)
            
        positions = []
        
        # Process non-USD balances
        for symbol, amount in account["balance"].items():
            if symbol != "USD" and amount > 0:
                # Get current price
                current_price = _get_current_price(f"{symbol}/USD")
                
                # Get entry price by checking transactions
                entry_transactions = [t for t in account["transactions"] 
                                     if t["symbol"] == symbol and t["type"] == "buy"]
                
                if entry_transactions:
                    # Calculate average entry price for the symbol
                    total_cost = sum(t["amount"] * t["price"] for t in entry_transactions)
                    total_amount = sum(t["amount"] for t in entry_transactions)
                    entry_price = total_cost / total_amount if total_amount > 0 else current_price
                    
                    # Calculate unrealized P&L
                    unrealized_pnl = (current_price - entry_price) * amount
                    pnl_percentage = (current_price / entry_price - 1) * 100;
                    
                    positions.append({
                        "symbol": f"{symbol}/USD",
                        "direction": "long",
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "size": amount,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_percentage": pnl_percentage
                    })
        
        return positions
        
    except Exception as e:
        print(f"Error getting position summary: {e}")
        return []

def _get_current_price(symbol):
    """Get current market price for a symbol with fallback to mock data"""
    try:
        # Try to get price from Alpaca API
        # For now, provide reasonable mock prices 
        # (removed Binance-specific code)
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Updated price map - removed references to Binance pricing
        price_map = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            # Removed BNB as it's primarily a Binance token
            "SOL": 150.0,
            "ADA": 0.5,
            "DOGE": 0.15,
        }
        
        base_price = price_map.get(base_symbol, 100.0)
        
        # Add small random variation
        import random
        variation = random.uniform(-0.01, 0.01)  # ±1%
        
        return base_price * (1 + variation)
        
    except Exception as e:
        print(f"Error getting current price: {e}")
        return 100.0  # Default fallback price

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
                # Handle Alpaca format (BTCUSD) - removed Binance format
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
    
    # Get risk settings from session state
    risk_settings = st.session_state.get("risk_settings", None)
    
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
        
        # Show recommended position size if available
        if risk_settings:
            recommended_size = risk_settings['position_size'] / _get_current_price(symbol)
            st.caption(f"Recommended size: {recommended_size:.4f} based on risk analysis")
            
        # Add stop loss and take profit details if available
        if risk_settings:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🛑 Stop Loss: ${risk_settings['stop_loss']:.2f}")
            with col2:
                st.caption(f"🎯 Take Profit: ${risk_settings['take_profit']:.2f}")
        
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
        
        # Execute the trade with risk settings
        result = execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price if order_type == "Limit" else None,
            order_type=order_type.lower(),
            risk_settings=risk_settings
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
