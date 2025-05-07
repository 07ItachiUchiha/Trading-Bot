import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.trade_controls import display_trade_controls
from dashboard.components.market_data import display_market_data
from dashboard.components.risk_management import display_risk_management_controls
from dashboard.components.wallet import load_account_data

def main():
    """Display the manual trading page"""
    st.title("📈 Manual Trading")
    st.write("Execute manual trades based on market data and trading signals.")
    
    # Get account data and symbol from session state
    account_data = load_account_data()
    symbol = st.session_state.get('symbol', 'BTC/USD')
    
    # Create a tab layout for better organization
    tab1, tab2 = st.tabs(["Trading Dashboard", "Risk Management"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display market data with combined signals
            display_market_data(symbol, show_combined_signals=True)
        
        with col2:
            # Display trade execution controls
            display_trade_controls(symbol)
    
    with tab2:
        # Get signal confidence from sentiment analysis (placeholder)
        # In a real implementation, this would come from your sentiment analysis component
        signal_confidence = st.session_state.get('signal_confidence', 0.7)
        
        # Display risk management controls
        risk_settings = display_risk_management_controls(
            symbol, 
            account_balance=account_data['balance'].get('USD', 10000),
            confidence=signal_confidence
        )
        
        # Store risk settings in session state for use by trade execution
        st.session_state.risk_settings = risk_settings
    
    # Display additional information
    with st.expander("About Manual Trading"):
        st.write("""
        Use this page to execute manual trades with real-time market data. 
        
        The system provides:
        - Real-time price charts
        - Technical indicators
        - News sentiment analysis
        - Trade execution controls
        - Position and order tracking
        
        **Risk Management Features:**
        - Intelligent position sizing based on confidence and risk tolerance
        - Automatic stop-loss calculation based on asset volatility
        - Take-profit targets with optimal risk:reward ratios
        - Trailing stops to lock in profits
        - Time-based exits for non-performing positions
        - Correlation protection to avoid overexposure
        
        Make informed trading decisions based on technical and fundamental analysis.
        """)

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # If run directly, set page config here
    st.set_page_config(
        page_title="Trading Bot - Manual Trading",
        page_icon="📊",
        layout="wide"
    )
    main()
