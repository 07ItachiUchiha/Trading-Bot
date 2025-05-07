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
    """Load account balance and transaction history"""
    # In a real implementation, this would fetch data from the exchange API
    # For demo purposes, we'll generate sample data
    
    # Account data structure
    account = {
        "balance": {
            "USD": 10000.0,
            "BTC": 0.1,
            "ETH": 1.5,
            "SOL": 10.0
        },
        "equity": 15000.0,  # Total value in USD
        "margin_used": 2000.0,
        "margin_available": 8000.0,
        "transactions": []
    }
    
    # Generate sample transaction history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, periods=20)
    
    for i, date in enumerate(dates):
        # Generate a mix of deposits, withdrawals, and trades
        if i < 3:
            # Initial deposits
            account["transactions"].append({
                "id": f"tx{i+1}",
                "type": "deposit",
                "amount": 3000 + i * 500,
                "currency": "USD",
                "timestamp": date.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "completed"
            })
        elif i >= 15:
            # Some withdrawals
            account["transactions"].append({
                "id": f"tx{i+1}",
                "type": "withdrawal",
                "amount": 500 + i * 100,
                "currency": "USD",
                "timestamp": date.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "completed"
            })
        else:
            # Trades
            is_buy = i % 2 == 0
            currency = ["BTC", "ETH", "SOL"][i % 3]
            price = 30000 if currency == "BTC" else 2000 if currency == "ETH" else 150
            amount = 0.01 if currency == "BTC" else 0.1 if currency == "ETH" else 1.0
            
            account["transactions"].append({
                "id": f"tx{i+1}",
                "type": "buy" if is_buy else "sell",
                "amount": amount,
                "currency": currency,
                "price": price * (1 + np.random.uniform(-0.05, 0.05)),
                "timestamp": date.strftime('%Y-%m-%d %H:%M:%S'),
                "status": "completed",
                "fee": amount * price * 0.002  # 0.2% fee
            })
    
    return account

def calculate_portfolio_value(balances, prices=None):
    """
    Calculate total portfolio value in USD
    
    Args:
        balances: Dictionary of currency balances
        prices: Dictionary of currency prices in USD (optional)
    """
    if prices is None:
        # Default prices if not provided
        prices = {
            "USD": 1.0,
            "BTC": 95000.0,
            "ETH": 3500.0,
            "SOL": 150.0,
            "DOGE": 0.15,
            "BNB": 500.0,
            "ADA": 0.5
        }
    
    total_value = 0.0
    portfolio_values = {}
    
    for currency, amount in balances.items():
        if currency in prices:
            value = amount * prices[currency]
            portfolio_values[currency] = value
            total_value += value
        else:
            # Default to 1:1 for unknown currencies
            portfolio_values[currency] = amount
            total_value += amount
    
    return total_value, portfolio_values

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
        ),
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
    )
    
    return fig

def plot_balance_history():
    """Create a line chart showing balance history over time"""
    # Generate sample balance history for demo purposes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Starting balance and daily changes
    starting_balance = 10000
    daily_changes = np.random.normal(0.005, 0.03, len(dates))  # Mean 0.5% daily return with 3% std
    
    # Calculate cumulative balance
    cumulative_factor = np.cumprod(1 + daily_changes)
    balances = starting_balance * cumulative_factor
    
    # Create dataframe
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
        labels={'balance': 'Balance (USD)', 'date': 'Date'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def display_wallet():
    """Display wallet and account information"""
    st.header("💰 Wallet & Account")
    
    # Load account data
    account = load_account_data()
    
    # Calculate portfolio value
    total_value, portfolio_values = calculate_portfolio_value(account["balance"])
    
    # Display account summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:.2f}")
    
    with col2:
        margin_used_pct = account["margin_used"] / account["equity"] * 100 if account["equity"] > 0 else 0
        st.metric("Margin Used", f"${account['margin_used']:.2f}", f"{margin_used_pct:.1f}%")
    
    with col3:
        st.metric("Available Cash", f"${account['balance']['USD']:.2f}")
    
    # Create tabs for different wallet views
    tab1, tab2, tab3 = st.tabs(["Portfolio", "Assets", "Transactions"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show portfolio allocation pie chart
            fig = plot_portfolio_breakdown(portfolio_values)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show balance history
            fig = plot_balance_history()
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Display individual assets and balances
        asset_data = []
        
        for currency, amount in account["balance"].items():
            if currency == "USD":
                price = 1.0
            else:
                # In a real implementation, fetch current prices from API
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
        transaction_data = account["transactions"]
        
        # Convert to DataFrame for display
        if transaction_data:
            df = pd.DataFrame(transaction_data)
            
            # Sort by timestamp (newest first)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
            
            # Display
            st.dataframe(
                df,
                column_config={
                    "id": st.column_config.TextColumn("Transaction ID"),
                    "type": st.column_config.TextColumn("Type"),
                    "amount": st.column_config.NumberColumn("Amount", format="%.4f"),
                    "currency": st.column_config.TextColumn("Currency"),
                    "price": st.column_config.NumberColumn("Price", format="%.2f"),
                    "timestamp": st.column_config.DatetimeColumn("Time"),
                    "status": st.column_config.TextColumn("Status"),
                    "fee": st.column_config.NumberColumn("Fee", format="%.4f")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transaction history available")
    
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
    
    # Show top 2 crypto holdings
    crypto_balances = {k: v for k, v in account["balance"].items() if k != "USD"}
    
    # Sort by assumed value
    crypto_values = []
    for currency, amount in crypto_balances.items():
        # Approximate prices for display
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
