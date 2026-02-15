import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

def load_account_data():
    """Load account balance and transactions from disk."""
    try:
        account_file = Path(__file__).parent.parent.parent / "data" / "account.json"
        
        # If file doesn't exist, create a default account structure
        if not account_file.exists():
            default_account = {
                "balance": {
                    "USD": 10000.0,
                    "BTC": 0.0,
                    "ETH": 0.0,
                    "SOL": 0.0,
                    "ADA": 0.0,
                    "DOGE": 0.0,
                    # Removed BNB as it's primarily a Binance token
                },
                "equity": 10000.0,
                "margin_used": 0.0,
                "transactions": []
            }
            
            # Make sure the directory exists
            account_file.parent.mkdir(exist_ok=True)
            
            # Write default account data
            with open(account_file, 'w') as f:
                json.dump(default_account, f, indent=2)
                
            return default_account
        
        # Load existing account data
        with open(account_file, 'r') as f:
            return json.load(f)
    
    except Exception as e:
        st.error(f"Error loading account data: {e}")
        # Return default structure if there's an error
        return {
            "balance": {"USD": 10000.0},
            "equity": 10000.0,
            "margin_used": 0.0,
            "transactions": []
        }

def calculate_portfolio_value(balances, prices=None):
    """Calculate total portfolio value with current market prices"""
    try:
        total_value = balances.get("USD", 0)
        portfolio_values = {"Cash (USD)": total_value}
        
        # Get current prices for non-USD assets
        for currency, amount in balances.items():
            if currency != "USD" and amount > 0:
                # Get current price
                if prices and currency in prices:
                    price = prices[currency]
                else:
                    # Fetch price from API or use mock values
                    price = _get_current_price(currency)
                
                value = amount * price
                total_value += value
                portfolio_values[currency] = value
        
        return total_value, portfolio_values
    
    except Exception as e:
        st.error(f"Error calculating portfolio value: {e}")
        return 10000.0, {"Cash (USD)": 10000.0}

def _get_current_price(symbol):
    """Get current market price for a symbol with fallback to mock data"""
    # Price map for mock data - added XAU/USD
    price_map = {
        "BTC": 65000.0,
        "ETH": 3500.0,
        "SOL": 150.0,
        "ADA": 0.5,
        "DOGE": 0.15,
        "XAU": 2400.0,
        "GOLD": 2400.0,
    }
    
    base_price = price_map.get(symbol, 100.0)
    
    # Add small random variation
    variation = np.random.uniform(-0.01, 0.01)  # ±1%
    
    return base_price * (1 + variation)

def refresh_account_data():
    """Force refresh of account data"""
    if 'account_last_refresh' in st.session_state:
        del st.session_state.account_last_refresh

def display_wallet():
    """Display wallet and account information"""
    st.header("Wallet & Account")
    
    # Check if we need to refresh data (every 30 seconds)
    if ('account_last_refresh' not in st.session_state or 
        (datetime.now() - st.session_state.account_last_refresh).total_seconds() > 30):
        st.session_state.account_data = load_account_data()
        st.session_state.account_last_refresh = datetime.now()
    
    # Use cached account data
    account = st.session_state.account_data
    
    # Calculate portfolio value
    total_value, portfolio_values = calculate_portfolio_value(account["balance"])
    
    # Add refresh button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh"):
            refresh_account_data()
            st.rerun()
    
    # Display account summary
    st.subheader("Account Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:.2f}")
    
    with col2:
        margin_used = account.get("margin_used", 0)
        margin_used_pct = margin_used / total_value * 100 if total_value > 0 else 0
        st.metric("Margin Used", f"${margin_used:.2f}", f"{margin_used_pct:.1f}%")
    
    with col3:
        st.metric("Available Cash", f"${account['balance'].get('USD', 0):.2f}")
    
    # Create tabs for different wallet views
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Assets", "Transactions", "Transfer Funds"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show portfolio allocation pie chart
            fig = plot_portfolio_breakdown(portfolio_values)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show balance history
            fig = plot_balance_history(account.get("transactions", []))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Display individual assets and balances
        asset_data = []
        
        for currency, amount in account["balance"].items():
            if currency == "USD":
                price = 1.0
            else:
                # TODO: pull live prices from the API instead of hardcoding
                price = 95000.0 if currency == "BTC" else 3500.0 if currency == "ETH" else 150.0
            
            asset_data.append({
                "Asset": currency,
                "Balance": amount,
                "Current Price": price,
                "Value (USD)": amount * price
            })
        
        assets_df = pd.DataFrame(asset_data)
        
        # Format for display
        formatted_df = assets_df.copy()
        formatted_df["Current Price"] = formatted_df["Current Price"].apply(lambda x: f"${x:.2f}")
        formatted_df["Value (USD)"] = formatted_df["Value (USD)"].apply(lambda x: f"${x:.2f}")
        
        # Show as table
        st.dataframe(
            formatted_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Balance": st.column_config.NumberColumn("Balance", format="%.6f"),
                "Current Price": st.column_config.TextColumn("Current Price"),
                "Value (USD)": st.column_config.TextColumn("Value (USD)")
            },
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        # Show transaction history
        transaction_data = account.get("transactions", [])
        
        # Convert to DataFrame for display
        if transaction_data:
            try:
                df = pd.DataFrame(transaction_data)
                
                # Sort by timestamp (newest first)
                if 'timestamp' in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                    df = df.dropna(subset=['timestamp'])  # Remove rows with invalid timestamps
                    df = df.sort_values("timestamp", ascending=False)
                
                # Display with flexible column configuration
                column_config = {col: st.column_config.TextColumn(col) for col in df.columns}
                
                # Override specific column configurations if they exist
                if 'id' in df.columns:
                    column_config['id'] = st.column_config.TextColumn("Transaction ID")
                if 'type' in df.columns:
                    column_config['type'] = st.column_config.TextColumn("Type")
                if 'amount' in df.columns:
                    column_config['amount'] = st.column_config.NumberColumn("Amount", format="%.4f")
                if 'price' in df.columns:
                    column_config['price'] = st.column_config.NumberColumn("Price", format="%.2f")
                if 'timestamp' in df.columns:
                    column_config['timestamp'] = st.column_config.DatetimeColumn("Time")
                if 'status' in df.columns:
                    column_config['status'] = st.column_config.TextColumn("Status")
                if 'fee' in df.columns:
                    column_config['fee'] = st.column_config.NumberColumn("Fee", format="%.4f")
                
                # Display the dataframe
                st.dataframe(
                    df,
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error displaying transactions: {e}")
                st.info("No valid transaction history available")
        else:
            st.info("No transaction history available")
    
    with tab4:
        # Provide deposit/withdraw options
        st.subheader("Transfer Funds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Deposit form
            with st.form("deposit_form"):
                st.write("Deposit Funds")
                deposit_amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
                deposit_currency = st.selectbox("Currency", ["USD", "BTC", "ETH", "SOL"])
                
                deposit_submitted = st.form_submit_button("Deposit")
                st.info("This feature is not yet implemented.")
                
            if deposit_submitted:
                st.success(f"Initiated deposit of {deposit_amount} {deposit_currency}. Please complete the process in your payment provider.")
        
        with col2:
            # Withdrawal form
            with st.form("withdraw_form"):
                st.write("Withdraw Funds")
                withdraw_amount = st.number_input("Amount", min_value=0.0, value=50.0, step=10.0)
                withdraw_currency = st.selectbox("Currency", ["USD", "BTC", "ETH", "SOL"])
                withdraw_address = st.text_input("Withdrawal Address/Account")
                
                withdraw_submitted = st.form_submit_button("Withdraw")
            
            if withdraw_submitted:
                if withdraw_address:
                    max_withdrawal = account["balance"].get(withdraw_currency, 0)
                    if withdraw_amount <= max_withdrawal:
                        st.success(f"Withdrawal request submitted for {withdraw_amount} {withdraw_currency} to {withdraw_address}")
                    else:
                        st.error(f"Insufficient balance. Maximum withdrawal amount is {max_withdrawal} {withdraw_currency}")
                else:
                    st.error("Please provide a valid withdrawal address")

def plot_portfolio_breakdown(portfolio_values):
    """Create a pie chart showing portfolio allocation"""
    labels = list(portfolio_values.keys())
    values = list(portfolio_values.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.4,
        textinfo='label+percent',
        marker=dict(
            colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
        )
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
    )
    
    return fig

def plot_balance_history(transactions=[]):
    """Create a line chart showing balance history over time"""
    # If no transactions, create some sample data
    if not transactions:
        # Generate sample balance history
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days).tolist()
        balance = 10000.0
        balances = [balance]
        
        for _ in range(1, days):
            change = np.random.normal(0, balance * 0.01)  # 1% standard deviation
            balance += change
            balances.append(max(balance, 5000))  # Ensure balance doesn't go too low
            
        df = pd.DataFrame({
            'date': dates,
            'balance': balances
        })
    else:
        try:
            # Process real transaction history
            start_balance = 10000.0  # Starting balance
            
            # Convert transactions to DataFrame
            df_transactions = pd.DataFrame(transactions)
            
            if not df_transactions.empty:
                # Ensure timestamp is datetime and handle errors
                if 'timestamp' in df_transactions.columns:
                    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'], errors='coerce')
                    # Drop rows with invalid timestamps
                    df_transactions = df_transactions.dropna(subset=['timestamp'])
                    
                    # Sort by timestamp
                    df_transactions = df_transactions.sort_values('timestamp')
                    
                    # Calculate running balance
                    daily_balances = []
                    balance = start_balance
                    
                    # Get unique dates
                    min_date = df_transactions['timestamp'].min().date()
                    dates = pd.date_range(
                        start=min_date,
                        end=datetime.now().date()
                    )
                    
                    for date in dates:
                        # Get transactions for this date
                        day_transactions = df_transactions[
                            (df_transactions['timestamp'].dt.date <= date.date())
                        ]
                        
                        # Calculate balance based on transactions
                        if not day_transactions.empty:
                            # If usd_value exists, use it; otherwise calculate from price and amount
                            if 'usd_value' in day_transactions.columns:
                                balance = start_balance + day_transactions['usd_value'].sum()
                            elif all(col in day_transactions.columns for col in ['price', 'amount', 'type']):
                                # Calculate the impact on balance
                                for _, t in day_transactions.iterrows():
                                    if t['type'] == 'buy':
                                        balance -= t['price'] * t['amount']
                                    elif t['type'] == 'sell':
                                        balance += t['price'] * t['amount']
                        
                        daily_balances.append({
                            'date': date,
                            'balance': balance
                        })
                    
                    df = pd.DataFrame(daily_balances)
                else:
                    # If no timestamp column, use mock data
                    raise KeyError("No timestamp column in transactions")
            else:
                # Fallback to sample data if transactions are empty
                raise ValueError("Empty transactions DataFrame")
                
        except Exception as e:
            print(f"Error processing transaction history: {e}")
            # Fallback to sample data if there's an error
            days = 30
            dates = pd.date_range(end=datetime.now(), periods=days).tolist()
            balances = [start_balance] * days
            df = pd.DataFrame({
                'date': dates,
                'balance': balances
            })
    
    # Create line chart
    fig = px.line(
        df, 
        x='date', 
        y='balance',
        title="Account Balance History",
        labels={'balance': 'Balance (USD)', 'date': 'Date'},
        markers=True
    )
    
    fig.update_layout(
        height=300,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    return fig

def display_quick_wallet():
    """Display a compact wallet summary for display in sidebars"""
    # Load account data
    account = load_account_data()
    
    # Calculate portfolio value
    total_value, _ = calculate_portfolio_value(account["balance"])
    
    # Display compact summary
    st.write("💰 **Wallet Summary**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio", f"${total_value:.2f}")
    with col2:
        st.metric("Cash", f"${account['balance']['USD']:.2f}")
    
    # Show top 2 crypto holdings - excluding Binance tokens
    crypto_balances = {k: v for k, v in account["balance"].items() if k != "USD" and k != "BNB"}
    
    # Sort by assumed value
    crypto_values = []
    for currency, amount in crypto_balances.items():
        # Approximate prices for display - removed Binance tokens
        if currency == "BTC":
            price = 95000
        elif currency == "ETH":
            price = 3500
        else:
            price = 150
        
        crypto_values.append((currency, amount, price * amount))
    
    # Sort by value and take top 2
    crypto_values.sort(key=lambda x: x[2], reverse=True)
    top_cryptos = crypto_values[:2]
    
    if top_cryptos:
        for currency, amount, _ in top_cryptos:
            st.text(f"{currency}: {amount:.6f}")
    
    # Add a "View Full Wallet" link
    st.write("[View Full Wallet](/Wallet)")

# If the file is run directly, display the wallet
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Trading Bot - Wallet", layout="wide")
    display_wallet()
