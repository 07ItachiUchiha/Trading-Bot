import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

def load_trade_history(filepath=None):
    """
    Load trade history from file or use sample data
    
    Args:
        filepath: Optional path to trade history JSON file
    
    Returns:
        DataFrame containing trade history
    """
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data.get('trades', []))
            
            # Parse dates
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
        except Exception as e:
            st.error(f"Error loading trade history: {e}")
    
    # If no file or error, create sample data
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
    """
    Plot cumulative PnL over time
    
    Args:
        trade_data: DataFrame containing trade history
        aggregate_by_day: Whether to aggregate PnL by day
        
    Returns:
        Plotly figure
    """
    if trade_data.empty:
        return go.Figure()
    
    # Make a copy of the data
    df = trade_data.copy()
    
    # Make sure all required columns exist
    required_cols = ['timestamp', 'pnl', 'action', 'symbol']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Required column {col} not found in trade data")
            return go.Figure()
    
    # Make sure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    if aggregate_by_day:
        # Add date column for aggregation
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate by day
        daily_pnl = df.groupby('date').agg({
            'pnl': 'sum',
            'timestamp': 'first'  # Keep first timestamp of the day
        }).reset_index()
        
        # Calculate cumulative PnL
        daily_pnl['cumulative_pnl'] = daily_pnl['pnl'].cumsum()
        
        # Create figure
        fig = px.line(
            daily_pnl, 
            x='date', 
            y='cumulative_pnl',
            title='Daily PnL Performance',
            labels={'cumulative_pnl': 'Cumulative PnL', 'date': 'Date'}
        )
        
        # Add markers for positive and negative days
        positive_days = daily_pnl[daily_pnl['pnl'] > 0]
        negative_days = daily_pnl[daily_pnl['pnl'] < 0]
        
        fig.add_trace(go.Scatter(
            x=positive_days['date'],
            y=positive_days['cumulative_pnl'],
            mode='markers',
            marker=dict(color='green', size=8),
            name='Positive Day'
        ))
        
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
        
        # Add Buy/Sell markers
        buy_trades = df[df['action'] == 'buy']
        sell_trades = df[df['action'] == 'sell']
        
        fig.add_trace(go.Scatter(
            x=buy_trades['timestamp'],
            y=buy_trades['cumulative_pnl'],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_trades['timestamp'],
            y=sell_trades['cumulative_pnl'],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ))
    
    # Add horizontal line at 0
    fig.add_shape(
        type='line',
        x0=df['timestamp'].min(),
        y0=0,
        x1=df['timestamp'].max(),
        y1=0,
        line=dict(color='gray', dash='dash'),
    )
    
    # Improve layout
    fig.update_layout(
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

def display_pnl_chart(trades_df=None):
    """Display PnL visualization in Streamlit"""
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
    fig = plot_pnl_chart(trades_df, aggregate_by_day)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    if not trades_df.empty:
        with st.expander("PnL Summary Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            
            total_pnl = trades_df['pnl'].sum()
            profitable_trades = (trades_df['pnl'] > 0).sum()
            losing_trades = (trades_df['pnl'] < 0).sum()
            win_rate = profitable_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            col1.metric("Total PnL", f"${total_pnl:.2f}")
            col2.metric("Profitable Trades", profitable_trades)
            col3.metric("Losing Trades", losing_trades)
            col4.metric("Win Rate", f"{win_rate:.2f}%")

def find_trade_history_files():
    """Find trade history files in the exports directory"""
    exports_dir = Path(__file__).parent.parent.parent / "exports"
    
    if not exports_dir.exists():
        return [None]
    
    files = list(exports_dir.glob("trade_history_*.json"))
    return [None] + [str(f) for f in files]  # None for sample data
