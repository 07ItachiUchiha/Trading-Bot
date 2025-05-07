import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import wallet component
from dashboard.components.wallet import display_wallet

# Page configuration
st.set_page_config(
    page_title="Trading Bot - Wallet",
    page_icon="💰",
    layout="wide"
)

# Display wallet page
display_wallet()

# Additional information
with st.expander("About Your Wallet"):
    st.write("""
    This page displays your trading account information:
    
    * **Portfolio Value**: Total value of all your assets in USD
    * **Assets**: Individual cryptocurrency and fiat balances
    * **Transactions**: History of your deposits, withdrawals, and trades
    * **Transfer Funds**: Options to deposit or withdraw assets
    
    Please note that all transaction data shown is for demonstration purposes only.
    In the real version, this would connect to your actual trading account.
    """)
