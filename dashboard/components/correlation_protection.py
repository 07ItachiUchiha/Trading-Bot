import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def display_correlation_matrix(symbols=None):
    """Show the correlation matrix heatmap and flag any risky overlaps."""
    st.subheader("Correlation Protection")
    
    if not symbols:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOGE/USD"]
    
    # Generate sample correlation data
    correlation_data = _generate_correlation_matrix(symbols)
    
    # Plot correlation matrix
    fig = px.imshow(
        correlation_data,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        labels=dict(x="Asset", y="Asset", color="Correlation"),
        zmin=-1, zmax=1
    )
    
    fig.update_layout(height=400, width=600)
    st.plotly_chart(fig)
    
    # Check for high correlations
    high_correlations = []
    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            if i < j:  # Avoid duplicates and self-correlations
                corr = correlation_data.iloc[i, j]
                if abs(corr) >= 0.7:
                    high_correlations.append((sym1, sym2, corr))
    
    # Display warnings for high correlations
    if high_correlations:
        st.warning("High Asset Correlations Detected")
        for sym1, sym2, corr in high_correlations:
            st.write(f"**{sym1}** and **{sym2}** have a correlation of **{corr:.2f}**")
        
        st.info("""
        **Risk Management Recommendation:**
        Avoid adding both assets to your portfolio as they move together.
        This could amplify both gains and losses.
        """)
    else:
        st.success("No concerning correlations detected between assets")
        
    # Display risk diversification recommendations
    st.subheader("Portfolio Diversification")
    st.write("""
    **Recommended Exposure Limits:**
    - Maximum 25% in any single asset
    - Maximum 40% in highly correlated assets (correlation > 0.7)
    - Minimum 20% in uncorrelated or negative correlated assets
    """)

def _generate_correlation_matrix(symbols):
    """Generate a realistic correlation matrix for demo purposes"""
    # Define realistic correlations
    base_correlations = {
        ("BTC/USD", "ETH/USD"): 0.82,
        ("BTC/USD", "SOL/USD"): 0.65,
        ("BTC/USD", "ADA/USD"): 0.61,
        ("BTC/USD", "DOGE/USD"): 0.58,
        ("ETH/USD", "SOL/USD"): 0.74,
        ("ETH/USD", "ADA/USD"): 0.69,
        ("ETH/USD", "DOGE/USD"): 0.52,
        ("SOL/USD", "ADA/USD"): 0.71,
        ("SOL/USD", "DOGE/USD"): 0.48,
        ("ADA/USD", "DOGE/USD"): 0.53,
    }
    
    # Create correlation matrix
    n = len(symbols)
    corr = pd.DataFrame(1.0, index=symbols, columns=symbols)  # Initialize with 1s
    
    for i, sym1 in enumerate(symbols):
        for j, sym2 in enumerate(symbols):
            if i != j:  # Skip diagonal
                # Try to find correlation in base_correlations
                key = (sym1, sym2) if (sym1, sym2) in base_correlations else (sym2, sym1)
                if key in base_correlations:
                    corr.iloc[i, j] = base_correlations[key]
                else:
                    # Generate a reasonable value
                    corr.iloc[i, j] = 0.3 + (np.random.random() * 0.4)
    
    return corr
