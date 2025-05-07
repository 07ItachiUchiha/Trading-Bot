import streamlit as st
import pandas as pd
from pathlib import Path
import json

def display_strategy_parameters():
    """Display strategy parameter configuration UI"""
    st.subheader("Strategy Parameters")
    
    # Load available strategies
    strategies = get_available_strategies()
    
    # Get selected strategy (either from session state or session)
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = list(strategies.keys())[0] if strategies else "No strategies available"
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Strategy",
        options=list(strategies.keys()),
        index=list(strategies.keys()).index(st.session_state.selected_strategy) if st.session_state.selected_strategy in strategies else 0
    )
    
    # Update session state with selected strategy
    st.session_state.selected_strategy = strategy
    
    # Get strategy details
    strategy_details = strategies.get(strategy, {})
    
    # Display strategy description
    if 'description' in strategy_details:
        st.info(strategy_details['description'])
    
    # Strategy parameter configuration
    st.subheader("Parameter Configuration")
    
    # Create a form to collect all parameters
    with st.form("strategy_parameters_form"):
        # Store updated parameters in this dict
        updated_params = {}
        
        # Get default parameters and their details
        default_params = strategy_details.get('parameters', {})
        
        # Group parameters into categories
        parameter_groups = {}
        
        # Assign each parameter to a group
        for param_name, param_details in default_params.items():
            group = param_details.get('group', 'General')
            if group not in parameter_groups:
                parameter_groups[group] = []
            parameter_groups[group].append((param_name, param_details))
        
        # Create tabs for each parameter group
        if parameter_groups:
            tabs = st.tabs(list(parameter_groups.keys()))
            
            for i, (group_name, params) in enumerate(parameter_groups.items()):
                with tabs[i]:
                    for param_name, param_details in params:
                        # Get parameter type and default value
                        param_type = param_details.get('type', 'float')
                        default_value = param_details.get('default', 0)
                        
                        # Get min, max, step if applicable
                        min_value = param_details.get('min', None)
                        max_value = param_details.get('max', None)
                        step = param_details.get('step', None)
                        options = param_details.get('options', None)
                        
                        # Get description/help
                        help_text = param_details.get('description', '')
                        
                        # Create the appropriate input widget based on parameter type
                        if param_type == 'float':
                            updated_params[param_name] = st.number_input(
                                f"{param_name}",
                                value=default_value,
                                min_value=min_value,
                                max_value=max_value,
                                step=step if step is not None else 0.1,
                                help=help_text
                            )
                        elif param_type == 'int':
                            updated_params[param_name] = st.number_input(
                                f"{param_name}",
                                value=int(default_value),
                                min_value=int(min_value) if min_value is not None else None,
                                max_value=int(max_value) if max_value is not None else None,
                                step=int(step) if step is not None else 1,
                                help=help_text
                            )
                        elif param_type == 'bool':
                            updated_params[param_name] = st.checkbox(
                                f"{param_name}",
                                value=bool(default_value),
                                help=help_text
                            )
                        elif param_type == 'select' and options:
                            updated_params[param_name] = st.selectbox(
                                f"{param_name}",
                                options=options,
                                index=options.index(default_value) if default_value in options else 0,
                                help=help_text
                            )
                        elif param_type == 'multiselect' and options:
                            updated_params[param_name] = st.multiselect(
                                f"{param_name}",
                                options=options,
                                default=default_value if isinstance(default_value, list) else [default_value],
                                help=help_text
                            )
                        else:
                            # Default to text input
                            updated_params[param_name] = st.text_input(
                                f"{param_name}",
                                value=str(default_value),
                                help=help_text
                            )
        else:
            st.info("This strategy has no configurable parameters")
        
        # Submit button to save parameters
        submitted = st.form_submit_button("Save Parameters")
        
        if submitted:
            save_strategy_parameters(strategy, updated_params)
            st.success(f"Parameters for strategy '{strategy}' have been saved")
    
    # Parameter templates
    with st.expander("Parameter Templates"):
        # Template selection
        templates = get_parameter_templates(strategy)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            template_name = st.selectbox(
                "Select Template",
                options=list(templates.keys()),
                index=0 if templates else 0
            )
        
        with col2:
            if st.button("Load Template") and template_name in templates:
                load_parameter_template(strategy, template_name)
                st.success(f"Template '{template_name}' loaded")
                st.rerun()
        
        # Template saving
        new_template_name = st.text_input("Save current parameters as template")
        
        if st.button("Save as Template") and new_template_name:
            save_parameter_template(strategy, new_template_name, updated_params)
            st.success(f"Template '{new_template_name}' saved")

    # Backtesting controls
    st.subheader("Backtest Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365))
    
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp.now())
    
    with col3:
        symbols = st.multiselect(
            "Symbols",
            options=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOGE/USD", "AAPL", "MSFT", "GOOGL"],
            default=["BTC/USD"]
        )
    
    # Run backtest button
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # In a real implementation, this would call your backtest engine
            st.session_state.show_results = True
            st.success("Backtest completed successfully!")
            st.rerun()

def get_available_strategies():
    """Get list of available strategies with their descriptions and parameters"""
    # This would typically load from a configuration file or database
    # For now, return hard-coded sample strategies
    return {
        "EMA Crossover Strategy": {
            "description": "A trend following strategy based on exponential moving average crossovers",
            "parameters": {
                "fast_ema": {"type": "int", "default": 12, "min": 2, "max": 50, "step": 1, "group": "General", "description": "Fast EMA period"},
                "slow_ema": {"type": "int", "default": 26, "min": 5, "max": 200, "step": 1, "group": "General", "description": "Slow EMA period"},
                "signal_line": {"type": "int", "default": 9, "min": 2, "max": 20, "step": 1, "group": "General", "description": "Signal line period"},
                "stop_loss_pct": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "group": "Risk Management", "description": "Stop-loss percentage"},
                "take_profit_pct": {"type": "float", "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5, "group": "Risk Management", "description": "Take profit percentage"},
                "position_size_pct": {"type": "float", "default": 5.0, "min": 1.0, "max": 100.0, "step": 1.0, "group": "Risk Management", "description": "Position size as percentage of portfolio"}
            }
        },
        "Bollinger Bands Strategy": {
            "description": "Mean reversion strategy using Bollinger Bands, RSI and volume",
            "parameters": {
                "bb_length": {"type": "int", "default": 20, "min": 10, "max": 50, "step": 1, "group": "Indicator Settings", "description": "Bollinger Bands period"},
                "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "group": "Indicator Settings", "description": "Number of standard deviations"},
                "rsi_length": {"type": "int", "default": 14, "min": 2, "max": 30, "step": 1, "group": "Indicator Settings", "description": "RSI period"},
                "rsi_oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "step": 1, "group": "Entry/Exit", "description": "RSI oversold level"},
                "rsi_overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "step": 1, "group": "Entry/Exit", "description": "RSI overbought level"},
                "volume_factor": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "group": "Entry/Exit", "description": "Volume factor for confirmation"},
                "use_trailing_stop": {"type": "bool", "default": True, "group": "Risk Management", "description": "Enable trailing stop"},
                "trailing_stop_pct": {"type": "float", "default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1, "group": "Risk Management", "description": "Trailing stop percentage"}
            }
        },
        "Breakout Strategy": {
            "description": "Volatility breakout strategy based on ATR and support/resistance",
            "parameters": {
                "atr_period": {"type": "int", "default": 14, "min": 5, "max": 30, "step": 1, "group": "Volatility", "description": "ATR calculation period"},
                "atr_multiplier": {"type": "float", "default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1, "group": "Volatility", "description": "ATR multiplier for breakout detection"},
                "consolidation_periods": {"type": "int", "default": 24, "min": 12, "max": 100, "step": 1, "group": "Pattern", "description": "Number of periods for consolidation"},
                "breakout_volume_factor": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1, "group": "Entry/Exit", "description": "Volume factor for breakout confirmation"},
                "pattern_type": {"type": "select", "default": "any", "options": ["any", "triangle", "rectangle", "flag"], "group": "Pattern", "description": "Type of consolidation pattern to look for"},
                "entry_delay": {"type": "int", "default": 1, "min": 0, "max": 5, "step": 1, "group": "Entry/Exit", "description": "Number of periods to delay entry after breakout"},
                "use_time_exit": {"type": "bool", "default": True, "group": "Risk Management", "description": "Enable time-based exit for non-performing trades"},
                "time_exit_periods": {"type": "int", "default": 12, "min": 1, "max": 50, "step": 1, "group": "Risk Management", "description": "Exit after this many periods if trade not profitable"}
            }
        },
        "Sentiment-Based Strategy": {
            "description": "Trading strategy based on market sentiment analysis from news and social media",
            "parameters": {
                "sentiment_threshold": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05, "group": "Sentiment", "description": "Minimum sentiment score to trigger signal"},
                "minimum_sources": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1, "group": "Sentiment", "description": "Minimum number of news sources required"},
                "sentiment_lookback": {"type": "int", "default": 24, "min": 1, "max": 72, "step": 1, "group": "Sentiment", "description": "Hours to look back for sentiment analysis"},
                "use_technical_confirmation": {"type": "bool", "default": True, "group": "Confirmation", "description": "Require technical confirmation for sentiment signals"},
                "technical_indicator": {"type": "select", "default": "macd", "options": ["macd", "rsi", "ema", "none"], "group": "Confirmation", "description": "Technical indicator for confirmation"},
                "use_volatility_filter": {"type": "bool", "default": True, "group": "Risk Management", "description": "Filter trades during high volatility"},
                "max_position_count": {"type": "int", "default": 5, "min": 1, "max": 20, "step": 1, "group": "Portfolio", "description": "Maximum number of concurrent positions"}
            }
        }
    }

def get_parameter_templates(strategy_name):
    """Get parameter templates for the given strategy"""
    # This would typically load from a configuration file or database
    # For now, return hard-coded sample templates
    templates = {
        "EMA Crossover Strategy": {
            "Default": {
                "fast_ema": 12,
                "slow_ema": 26,
                "signal_line": 9,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0,
                "position_size_pct": 5.0
            },
            "Conservative": {
                "fast_ema": 16,
                "slow_ema": 32,
                "signal_line": 10,
                "stop_loss_pct": 1.5,
                "take_profit_pct": 3.0,
                "position_size_pct": 2.0
            },
            "Aggressive": {
                "fast_ema": 8,
                "slow_ema": 21,
                "signal_line": 5,
                "stop_loss_pct": 3.0,
                "take_profit_pct": 6.0,
                "position_size_pct": 10.0
            }
        },
        "Bollinger Bands Strategy": {
            "Default": {
                "bb_length": 20,
                "bb_std": 2.0,
                "rsi_length": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "volume_factor": 1.5,
                "use_trailing_stop": True,
                "trailing_stop_pct": 1.0
            },
            "Tight Bands": {
                "bb_length": 15,
                "bb_std": 1.5,
                "rsi_length": 10,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "volume_factor": 1.2,
                "use_trailing_stop": True,
                "trailing_stop_pct": 0.8
            }
        }
    }
    
    return templates.get(strategy_name, {})

def save_strategy_parameters(strategy_name, parameters):
    """Save strategy parameters to persistent storage"""
    # In a real implementation, this would save to a database or file
    # For now, simply store in session state
    if 'saved_parameters' not in st.session_state:
        st.session_state.saved_parameters = {}
    
    st.session_state.saved_parameters[strategy_name] = parameters

def load_parameter_template(strategy_name, template_name):
    """Load parameters from a template"""
    templates = get_parameter_templates(strategy_name)
    if template_name in templates:
        save_strategy_parameters(strategy_name, templates[template_name])
        return True
    return False

def save_parameter_template(strategy_name, template_name, parameters):
    """Save current parameters as a template"""
    # In a real implementation, this would save to a database or file
    # For now, simply acknowledge the action
    pass

if __name__ == "__main__":
    # Test the component
    st.set_page_config(page_title="Strategy Parameters Test", layout="wide")
    display_strategy_parameters()
