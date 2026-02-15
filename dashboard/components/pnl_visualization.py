import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import numpy as np

def load_trade_history(filepath=None):
    """Load trade data from a JSON file, or generate sample data if none given."""
    try:
        if not filepath:
            # Use sample data if no file is provided
            return generate_sample_trade_data()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'trades' in data and data['trades']:
            df = pd.DataFrame(data['trades'])
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure we have numeric PnL data - replace any non-numeric values
            if 'pnl' in df.columns:
                df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
                
            return df
        else:
            return generate_sample_trade_data()
    except Exception as e:
        st.error(f"Error loading trade history: {e}")
        return generate_sample_trade_data()

def generate_sample_trade_data():
    """Generate sample trade data for demonstration"""
    # Create date range for the past month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, periods=50)
    
    # Symbols to use
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AAPL', 'MSFT', 'GOOGL']
    
    # Generate trade data
    data = []
    running_id = 1
    
    # Current positions to track for exit prices
    positions = {}
    
    for i, date in enumerate(dates):
        # Randomly choose to make a trade
        if i % 2 == 0:  # Every other date to ensure some trades
            symbol = symbols[i % len(symbols)]
            
            # Alternate between buy and sell, or close positions
            if symbol in positions:
                # Close position
                entry = positions[symbol]
                exit_price = entry['price'] * (1 + (0.05 * (-1 if entry['action'] == 'sell' else 1)))
                pnl = (exit_price - entry['price']) * entry['size'] * (-1 if entry['action'] == 'sell' else 1)
                
                data.append({
                    'id': running_id,
                    'symbol': symbol,
                    'action': 'sell' if entry['action'] == 'buy' else 'buy',  # Opposite of entry
                    'price': exit_price,
                    'size': entry['size'],
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'pnl': pnl,
                    'status': 'closed',
                    'entry_id': entry['id']
                })
                del positions[symbol]
                running_id += 1
            else:
                # New position
                action = 'buy' if i % 4 < 2 else 'sell'
                price = 100 * (i % 10 + 1) + (10 * (i % 7))
                size = 0.1 * (i % 5 + 1)
                
                entry_id = running_id
                data.append({
                    'id': entry_id,
                    'symbol': symbol,
                    'action': action,
                    'price': price,
                    'size': size,
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'pnl': 0,
                    'status': 'open',
                    'entry_id': None
                })
                
                # Store position
                positions[symbol] = {
                    'action': action,
                    'price': price,
                    'size': size,
                    'id': entry_id
                }
                running_id += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def plot_pnl_chart(trade_data, aggregate_by_day=False):
    """Build the PnL chart, handling edge cases like inf values."""
    # Convert to DataFrame if it's not already
    df = pd.DataFrame(trade_data)
    
    # Basic validation
    if df.empty:
        st.warning("No trade data available to plot")
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Trade Data Available",
            annotations=[dict(
                text="No trade data to display. Please execute some trades first.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return fig
        
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now()
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
    # Ensure PnL column exists with valid numeric values
    if 'pnl' not in df.columns:
        df['pnl'] = 0.0
    
    # Convert PnL to numeric, handle errors by setting to 0
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
    
    # Calculate cumulative PnL with handling for invalid values
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    # Replace any potential infinite values with NaN and then drop them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['cumulative_pnl'])
    
    if df.empty:
        st.warning("No valid trade data after filtering")
        fig = go.Figure()
        fig.update_layout(title="No Valid Trade Data")
        return fig
        
    # Now proceed with the charting logic
    if aggregate_by_day:
        # Add date column for aggregation
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate by day
        daily_pnl = df.groupby('date').agg({
            'pnl': 'sum',
            'timestamp': 'first'  # Keep first timestamp of the day
        }).reset_index()
        
        # Calculate cumulative PnL - safely
        daily_pnl['cumulative_pnl'] = daily_pnl['pnl'].cumsum()
        
        # Handle any infinite values
        daily_pnl.replace([np.inf, -np.inf], np.nan, inplace=True)
        daily_pnl = daily_pnl.dropna(subset=['cumulative_pnl'])
        
        # Create figure
        fig = px.line(
            daily_pnl, 
            x='date', 
            y='cumulative_pnl',
            title='Daily PnL Performance',
            labels={'cumulative_pnl': 'Cumulative PnL', 'date': 'Date'}
        )
        
        # Add markers for positive and negative days - safely
        positive_days = daily_pnl[daily_pnl['pnl'] > 0]
        negative_days = daily_pnl[daily_pnl['pnl'] < 0]
        
        if not positive_days.empty:
            fig.add_trace(go.Scatter(
                x=positive_days['date'],
                y=positive_days['cumulative_pnl'],
                mode='markers',
                marker=dict(color='green', size=8),
                name='Positive Day'
            ))
        
        if not negative_days.empty:
            fig.add_trace(go.Scatter(
                x=negative_days['date'],
                y=negative_days['cumulative_pnl'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Negative Day'
            ))
    else:
        # Basic line chart with all trades
        fig = px.line(
            df, 
            x='timestamp', 
            y='cumulative_pnl',
            title='Cumulative PnL Performance',
            labels={'cumulative_pnl': 'Cumulative PnL', 'timestamp': 'Time'}
        )
        
        # Add Buy/Sell markers if columns exist
        if 'action' in df.columns:
            buy_trades = df[df['action'] == 'buy']
            sell_trades = df[df['action'] == 'sell']
            
            if not buy_trades.empty:
                fig.add_trace(go.Scatter(
                    x=buy_trades['timestamp'],
                    y=buy_trades['cumulative_pnl'],
                    mode='markers',
                    marker=dict(color='blue', size=8, symbol='triangle-up'),
                    name='Buy'
                ))
            
            if not sell_trades.empty:
                fig.add_trace(go.Scatter(
                    x=sell_trades['timestamp'],
                    y=sell_trades['cumulative_pnl'],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    name='Sell'
                ))
    
    # Improve figure layout with better formatting and error prevention
    fig.update_layout(
        autosize=True,
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor='right',
            x=1
        ),
        yaxis=dict(
            title='Cumulative PnL ($)',
            tickprefix='$',
            fixedrange=False
        )
    )
    
    return fig

def display_pnl_chart(trades_df=None):
    """Render the PnL chart in the Streamlit UI."""
    st.header("📈 PnL Performance")
    
    with st.expander("PnL Chart Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to load different trade history files
            history_files = find_trade_history_files()
            selected_file = st.selectbox(
                "Select Trade History File", 
                options=history_files,
                format_func=lambda x: os.path.basename(x) if x else "Sample Data"
            )
        
        with col2:
            # Option to aggregate by day
            aggregate_by_day = st.checkbox("Aggregate by Day", value=False)
    
    # Load trade history
    if trades_df is None:
        trades_df = load_trade_history(selected_file)
    
    # Display chart
    try:
        fig = plot_pnl_chart(trades_df, aggregate_by_day)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        if not trades_df.empty and 'pnl' in trades_df.columns:
            with st.expander("PnL Summary Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                
                # Convert PnL to numeric safely
                pnl_values = pd.to_numeric(trades_df['pnl'], errors='coerce')
                
                # Calculate metrics safely
                total_pnl = pnl_values.sum()
                profitable_trades = (pnl_values > 0).sum()
                losing_trades = (pnl_values < 0).sum()
                win_rate = profitable_trades / len(pnl_values) * 100 if len(pnl_values) > 0 else 0
                
                col1.metric("Total PnL", f"${total_pnl:.2f}")
                col2.metric("Profitable Trades", profitable_trades)
                col3.metric("Losing Trades", losing_trades)
                col4.metric("Win Rate", f"{win_rate:.2f}%")
    except Exception as e:
        st.error(f"Error displaying PnL chart: {e}")
        st.info("Unable to render chart due to data issues. Please check your trade history data.")

def find_trade_history_files():
    """Find trade history files in the exports directory"""
    exports_dir = Path(__file__).parent.parent.parent / "exports"
    
    if not exports_dir.exists():
        return [None]
    
    files = list(exports_dir.glob("trade_history_*.json"))
    return [None] + [str(f) for f in files]  # None for sample data
