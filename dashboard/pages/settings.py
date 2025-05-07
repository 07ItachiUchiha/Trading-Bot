import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.bot_settings import display_bot_settings
from dashboard.components.api_settings import display_api_settings

def display_settings_page():
    """Display the settings page"""
    st.title("⚙️ Settings")
    st.write("Configure trading bot parameters and API connections.")
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["Bot Settings", "API Connections", "Notifications"])
    
    with tab1:
        display_bot_settings()
    
    with tab2:
        display_api_settings()
    
    with tab3:
        st.write("Configure notification settings")
        # Notification settings component would go here
    
    # Additional content goes here...

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - Settings",
        page_icon="⚙️",
        layout="wide"
    )
    display_settings_page()
else:
    # When imported, just call the display function without setting page config
    main = display_settings_page
