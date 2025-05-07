import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# REMOVE THIS - Don't set page config at the top level:
# st.set_page_config(
#     page_title="Trading Bot - Page Name",
#     page_icon="📊",
#     layout="wide"
# )

def display_page():
    """Display the page content"""
    st.title("Page Title")
    st.write("Page description...")
    
    # Page content here...

# This allows the file to be run directly or imported
if __name__ == "__main__":
    # Only set page config when running the file directly
    st.set_page_config(
        page_title="Trading Bot - Page Name",
        page_icon="📊",
        layout="wide"
    )
    display_page()
