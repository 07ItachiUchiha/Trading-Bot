import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

def display_backtest_results(strategy_name=None, backtest_data=None):
    """
    Display backtest results with interactive charts and metrics
    
    Args:
        strategy_name (str): Name of the strategy that was backtested
        backtest_data (dict): Backtest results data, if None will load sample data
    """
    if backtest_data is None:
        backtest_data = load_sample_backtest_data()
    
    # Use a passed strategy name or get from the data
    if strategy_name is None:
        strategy_name = backtest_data.get('strategy_name', 'Unnamed Strategy')
    
    st.header(f"Backtest Results: {strategy_name}")
    
    # Key performance metrics
    performance_metrics = backtest_data.get('performance', {})
    
    # Display summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = performance_metrics.get('total_return', 0)
        st.metric("Total Return", f"{total_return:.2f}%", 
                 delta=None if total_return == 0 else "↑" if total_return > 0 else "↓")
        
    with col2:
        win_rate = performance_metrics.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.2f}%")
        
    with col3:
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
    with col4:
        max_dd = performance_metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Equity Curve", "Trade Analysis", "Drawdown Analysis"])
    
    with tab1:
        # Display equity curve
        display_equity_curve(backtest_data)
        
    with tab2:
        # Display trade analysis
        display_trade_analysis(backtest_data)
        
    with tab3:
        # Display drawdown analysis
        display_drawdown_analysis(backtest_data)
    
    # Display detailed performance statistics
    with st.expander("Detailed Performance Statistics"):
        display_performance_table(performance_metrics)
    
    # Parameter settings used in the backtest
    with st.expander("Strategy Parameters"):
        parameters = backtest_data.get('parameters', {})
        
        for param, value in parameters.items():
            st.text(f"{param}: {value}")

def load_sample_backtest_data():
    """Load or generate sample backtest data for demonstration"""
    # Create simulated backtest data
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Create an equity curve with some random movements
    np.random.seed(42)  # For reproducibility
    initial_equity = 10000
    daily_returns = np.random.normal(0.001, 0.01, len(dates))  # Mean 0.1%, std 1%
    
    # Add a slight positive bias
    daily_returns += 0.0005
    
    # Calculate cumulative equity
    equity_curve = [initial_equity]
    for ret in daily_returns:
        equity_curve.append(equity_curve[-1] * (1 + ret))
    
    equity_curve = equity_curve[1:]  # Remove the initial value
    
    # Generate trades
    num_trades = 50
    trade_indices = np.sort(np.random.choice(range(len(dates)), num_trades, replace=False))
    
    trades = []
    for i, idx in enumerate(trade_indices):
        # Determine if winning trade (60% win rate)
        is_win = np.random.random() < 0.6
        
        # Generate reasonable PnL values
        if is_win:
            pnl_pct = np.random.uniform(0.5, 3.0)  # 0.5% to 3% win
        else:
            pnl_pct = -np.random.uniform(0.3, 2.0)  # 0.3% to 2% loss
            
        # Create trade object
        trades.append({
            'trade_id': i + 1,
            'entry_date': dates[max(0, idx-3)].strftime('%Y-%m-%d'),
            'exit_date': dates[idx].strftime('%Y-%m-%d'),
            'symbol': np.random.choice(['BTC/USD', 'ETH/USD', 'SOL/USD']),
            'direction': np.random.choice(['long', 'short']),
            'entry_price': np.random.uniform(10000, 60000),
            'exit_price': 0,  # To be calculated
            'size': np.random.uniform(0.1, 1.0),
            'pnl_pct': pnl_pct,
            'pnl_amount': 0,  # To be calculated
            'duration': np.random.randint(1, 14)  # 1-14 days
        })
        
        # Calculate exit price and pnl amount
        for trade in trades:
            if trade['direction'] == 'long':
                trade['exit_price'] = trade['entry_price'] * (1 + trade['pnl_pct']/100)
            else:
                trade['exit_price'] = trade['entry_price'] * (1 - trade['pnl_pct']/100)
                
            trade['pnl_amount'] = trade['size'] * (trade['exit_price'] - trade['entry_price'])
            if trade['direction'] == 'short':
                trade['pnl_amount'] *= -1
    
    # Calculate performance metrics
    total_return = ((equity_curve[-1] / equity_curve[0]) - 1) * 100
    profitable_trades = len([t for t in trades if t['pnl_amount'] > 0])
    win_rate = (profitable_trades / len(trades)) * 100 if trades else 0
    
    # Calculate Sharpe ratio
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Prepare sample backtest data
    return {
        'strategy_name': 'Sample EMA Crossover Strategy',
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'equity_curve': equity_curve,
        'trades': trades,
        'performance': {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': 1.8,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sharpe * 1.2,  # Just an approximation
            'max_drawdown': max_dd,
            'avg_trade': sum(t['pnl_amount'] for t in trades) / len(trades) if trades else 0,
            'avg_win': sum(t['pnl_amount'] for t in trades if t['pnl_amount'] > 0) / profitable_trades if profitable_trades else 0,
            'avg_loss': sum(t['pnl_amount'] for t in trades if t['pnl_amount'] <= 0) / (len(trades) - profitable_trades) if (len(trades) - profitable_trades) > 0 else 0,
            'total_trades': len(trades),
            'winning_trades': profitable_trades,
            'losing_trades': len(trades) - profitable_trades,
            'avg_trade_duration': sum(t['duration'] for t in trades) / len(trades) if trades else 0
        },
        'parameters': {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_line': 9,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'position_size_pct': 5.0
        }
    }

def display_equity_curve(backtest_data):
    """Display equity curve chart"""
    dates = backtest_data.get('dates', [])
    equity_curve = backtest_data.get('equity_curve', [])
    
    if not dates or not equity_curve or len(dates) != len(equity_curve):
        st.warning("Insufficient data to display equity curve")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'equity': equity_curve
    })
    
    # Create figure
    fig = px.line(
        df, 
        x='date', 
        y='equity',
        title='Equity Curve',
        labels={'equity': 'Equity ($)', 'date': 'Date'}
    )
    
    # Add reference line for initial equity
    fig.add_hline(
        y=equity_curve[0], 
        line_dash="dash", 
        line_color="gray", 
        annotation_text="Initial Equity"
    )
    
    # Add markers for trades if available
    trades = backtest_data.get('trades', [])
    if trades:
        # Extract trade exit points
        trade_exits = pd.DataFrame([
            {
                'date': pd.to_datetime(t['exit_date']),
                'equity': float(df[df['date'] == pd.to_datetime(t['exit_date'])]['equity'].values[0]) if pd.to_datetime(t['exit_date']) in df['date'].values else None,
                'pnl': t['pnl_amount'],
                'type': 'win' if t['pnl_amount'] > 0 else 'loss'
            }
            for t in trades if pd.to_datetime(t['exit_date']) in df['date'].values
        ])
        
        if not trade_exits.empty:
            # Add win trades
            wins = trade_exits[trade_exits['type'] == 'win']
            if not wins.empty:
                fig.add_trace(
                    go.Scatter(
                        x=wins['date'],
                        y=wins['equity'],
                        mode='markers',
                        marker=dict(color='green', size=8, symbol='circle'),
                        name='Win Trade'
                    )
                )
                
            # Add loss trades
            losses = trade_exits[trade_exits['type'] == 'loss']
            if not losses.empty:
                fig.add_trace(
                    go.Scatter(
                        x=losses['date'],
                        y=losses['equity'],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='circle'),
                        name='Loss Trade'
                    )
                )
    
    st.plotly_chart(fig, use_container_width=True)

def display_trade_analysis(backtest_data):
    """Display trade analysis charts"""
    trades = backtest_data.get('trades', [])
    
    if not trades:
        st.warning("No trade data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    with col1:
        # Create distribution of trade PnL
        fig1 = px.histogram(
            trades_df, 
            x='pnl_amount',
            title='Distribution of Trade PnL',
            labels={'pnl_amount': 'PnL ($)'},
            color_discrete_sequence=['lightblue']
        )
        
        # Add a vertical line at zero
        fig1.add_vline(x=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Create trade PnL by symbol
        fig2 = px.bar(
            trades_df.groupby('symbol').agg({'pnl_amount': 'sum'}).reset_index(),
            x='symbol',
            y='pnl_amount',
            title='PnL by Symbol',
            labels={'pnl_amount': 'Total PnL ($)', 'symbol': 'Symbol'},
            color='pnl_amount',
            color_continuous_scale=['red', 'green']
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Display trade details table
    with st.expander("Trade Details"):
        # Format trades for display
        display_df = trades_df[['trade_id', 'symbol', 'direction', 'entry_date', 'exit_date', 
                               'entry_price', 'exit_price', 'size', 'pnl_amount', 'duration']]
        
        # Format columns
        display_df['entry_price'] = display_df['entry_price'].map('${:,.2f}'.format)
        display_df['exit_price'] = display_df['exit_price'].map('${:,.2f}'.format)
        display_df['pnl_amount'] = display_df['pnl_amount'].map('${:,.2f}'.format)
        
        st.dataframe(display_df, use_container_width=True)

def display_drawdown_analysis(backtest_data):
    """Display drawdown analysis charts"""
    dates = backtest_data.get('dates', [])
    equity_curve = backtest_data.get('equity_curve', [])
    
    if not dates or not equity_curve or len(dates) != len(equity_curve):
        st.warning("Insufficient data to display drawdown analysis")
        return
    
    # Calculate underwater equity (drawdown) curve
    underwater = []
    peak = equity_curve[0]
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        underwater_pct = (peak - equity) / peak * 100
        underwater.append(underwater_pct)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'drawdown_pct': underwater
    })
    
    # Plot underwater equity chart
    fig = px.area(
        df,
        x='date',
        y='drawdown_pct',
        title='Drawdown Analysis',
        labels={'drawdown_pct': 'Drawdown (%)', 'date': 'Date'}
    )
    
    # Invert y-axis for better visualization
    fig.update_yaxes(autorange="reversed")
    
    # Add color fill to show depth of drawdown
    fig.update_traces(
        fillcolor="rgba(231,107,107,0.5)",
        line_color="rgb(231,107,107)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_performance_table(performance_metrics):
    """Display detailed performance metrics in a formatted table"""
    if not performance_metrics:
        st.warning("No performance metrics available")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Column 1: Return metrics
    with col1:
        st.subheader("Return Metrics")
        metrics1 = {
            "Total Return (%)": performance_metrics.get('total_return', 0),
            "Annual Return (%)": performance_metrics.get('annual_return', 0),
            "Sharpe Ratio": performance_metrics.get('sharpe_ratio', 0),
            "Sortino Ratio": performance_metrics.get('sortino_ratio', 0),
            "Profit Factor": performance_metrics.get('profit_factor', 0),
            "Max Drawdown (%)": performance_metrics.get('max_drawdown', 0)
        }
        
        # Format and display
        for name, value in metrics1.items():
            st.text(f"{name}: {value:.2f}")
    
    # Column 2: Trade metrics
    with col2:
        st.subheader("Trade Metrics")
        metrics2 = {
            "Total Trades": performance_metrics.get('total_trades', 0),
            "Win Rate (%)": performance_metrics.get('win_rate', 0),
            "Average Trade ($)": performance_metrics.get('avg_trade', 0),
            "Average Win ($)": performance_metrics.get('avg_win', 0),
            "Average Loss ($)": performance_metrics.get('avg_loss', 0),
            "Avg Duration (days)": performance_metrics.get('avg_trade_duration', 0)
        }
        
        # Format and display
        for name, value in metrics2.items():
            if name in ["Total Trades", "Winning Trades", "Losing Trades"]:
                st.text(f"{name}: {int(value)}")
            else:
                st.text(f"{name}: {value:.2f}")

if __name__ == "__main__":
    # Test the component
    st.set_page_config(page_title="Backtest Visualizer Test", layout="wide")
    display_backtest_results()
