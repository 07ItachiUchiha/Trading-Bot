import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import base64

def display_trade_filter(trades_df=None):
    """Display trade filtering system in Streamlit"""
    st.header("🔍 Trade Filtering System")
    
    # Load trade history if not provided
    if trades_df is None:
        from dashboard.components.pnl_visualization import load_trade_history, find_trade_history_files
        
        # Option to load different trade history files
        history_files = find_trade_history_files()
        selected_file = st.selectbox(
            "Select Trade History File", 
            options=history_files,
            format_func=lambda x: os.path.basename(x) if x else "Sample Data",
            key="filter_file_select"
        )
        
        trades_df = load_trade_history(selected_file)
    
    # Return if no data
    if trades_df.empty:
        st.info("No trade data available")
        return
    
    # Make sure timestamp is datetime
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    else:
        st.error("Timestamp column not found in trade data")
        return
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        min_date = trades_df['timestamp'].min().date()
        max_date = trades_df['timestamp'].max().date()
        
        # Use selectbox for predefined ranges to make it user-friendly
        date_range_options = {
            'All Time': (min_date, max_date),
            'Last 7 Days': (max_date - timedelta(days=7), max_date),
            'Last 30 Days': (max_date - timedelta(days=30), max_date),
            'Custom': None
        }
        
        date_range_selection = st.selectbox(
            "Date Range", 
            options=list(date_range_options.keys()),
            index=0
        )
        
        if date_range_selection == 'Custom':
            # Custom date range
            date_range = st.date_input(
                "Select Custom Date Range",
                value=(max_date - timedelta(days=7), max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
        else:
            start_date, end_date = date_range_options[date_range_selection]
    
    with col2:
        # Symbol filter
        symbols = ['All'] + sorted(trades_df['symbol'].unique().tolist())
        selected_symbol = st.selectbox("Symbol", symbols)
    
    with col3:
        # Action filter (Buy/Sell)
        actions = ['All'] + sorted(trades_df['action'].unique().tolist())
        selected_action = st.selectbox("Action", actions)
    
    # Search filter
    search_query = st.text_input("Search", placeholder="Search in trades...")
    
    # Apply filters
    filtered_df = trades_df.copy()
    
    # Date filter
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) & 
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Symbol filter
    if selected_symbol != 'All':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
    
    # Action filter
    if selected_action != 'All':
        filtered_df = filtered_df[filtered_df['action'] == selected_action]
    
    # Search
    if search_query:
        # Convert all columns to string and search
        mask = pd.Series(False, index=filtered_df.index)
        for col in filtered_df.columns:
            mask |= filtered_df[col].astype(str).str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Display filtered trades
    st.subheader("Filtered Trades")
    st.write(f"Showing {len(filtered_df)} of {len(trades_df)} trades")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Chart View"])
    
    with tab1:
        # Table with formatting
        if not filtered_df.empty:
            display_df = filtered_df.copy()
            
            # Format timestamp
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format PnL
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
            
            # Format price
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
            
            # Display table
            st.dataframe(
                display_df,
                column_config={
                    "action": st.column_config.TextColumn(
                        "Action",
                        help="Buy or Sell",
                        width="small",
                    ),
                    "symbol": st.column_config.TextColumn(
                        "Symbol",
                        width="small",
                    ),
                },
                use_container_width=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trade_data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No trades match your filters")
    
    with tab2:
        # Chart view of filtered trades
        if not filtered_df.empty:
            # Ensure numeric PnL
            filtered_df['pnl'] = pd.to_numeric(filtered_df['pnl'], errors='coerce').fillna(0)
            
            # Calculate cumulative PnL
            filtered_df = filtered_df.sort_values('timestamp')
            filtered_df['cumulative_pnl'] = filtered_df['pnl'].cumsum()
            
            # Plot
            fig = px.line(
                filtered_df, 
                x='timestamp', 
                y='cumulative_pnl',
                title='PnL for Filtered Trades',
                color='symbol' if selected_symbol == 'All' else None,
                labels={'cumulative_pnl': 'Cumulative PnL', 'timestamp': 'Time'}
            )
            
            # Add markers for buy/sell
            if selected_action == 'All' or selected_action == 'buy':
                buy_trades = filtered_df[filtered_df['action'] == 'buy']
                if not buy_trades.empty:
                    fig.add_scatter(
                        x=buy_trades['timestamp'],
                        y=buy_trades['cumulative_pnl'],
                        mode='markers',
                        marker=dict(color='green', symbol='triangle-up', size=10),
                        name='Buy'
                    )
            
            if selected_action == 'All' or selected_action == 'sell':
                sell_trades = filtered_df[filtered_df['action'] == 'sell']
                if not sell_trades.empty:
                    fig.add_scatter(
                        x=sell_trades['timestamp'],
                        y=sell_trades['cumulative_pnl'],
                        mode='markers',
                        marker=dict(color='red', symbol='triangle-down', size=10),
                        name='Sell'
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_pnl = filtered_df['pnl'].sum()
            profitable_trades = (filtered_df['pnl'] > 0).sum()
            losing_trades = (filtered_df['pnl'] < 0).sum()
            win_rate = profitable_trades / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            
            col1.metric("Total PnL", f"${total_pnl:.2f}")
            col2.metric("Profitable Trades", profitable_trades)
            col3.metric("Losing Trades", losing_trades)
            col4.metric("Win Rate", f"{win_rate:.2f}%")
        else:
            st.info("No trades match your filters")

def generate_download_link(df, filename):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href
