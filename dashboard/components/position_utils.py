import streamlit as st
from datetime import datetime

def register_position_with_monitor(trade_data):
    """Hook a new trade into the position monitor for automated exit management."""
    # Ensure position_monitor exists in session state
    if 'position_monitor' not in st.session_state:
        from dashboard.components.position_monitor import PositionMonitor
        st.session_state.position_monitor = PositionMonitor()
    
    monitor = st.session_state.position_monitor
    
    # Only register 'buy' trades (opening positions)
    if trade_data['action'].lower() == 'buy':
        position_data = {
            'id': trade_data['id'],
            'symbol': trade_data['symbol'],
            'direction': 'long',
            'entry_price': trade_data['price'],
            'size': trade_data['size'],
            'entry_time': datetime.now().isoformat(),
            'status': 'open'
        }
        
        # Add risk management settings
        if 'stop_loss' in trade_data:
            position_data['stop_loss'] = trade_data['stop_loss']
            
        if 'take_profit' in trade_data:
            position_data['take_profit'] = trade_data['take_profit']
            
        if trade_data.get('trailing_stop'):
            position_data['use_trailing_stop'] = True
            position_data['trailing_stop_percentage'] = trade_data.get('trailing_stop_percentage', 2.0)
            
        if trade_data.get('time_exit'):
            position_data['use_time_exit'] = True
            position_data['exit_time'] = trade_data.get('exit_time')
        
        # max loss exit settings
        if trade_data.get('use_max_loss_exit'):
            position_data['use_max_loss_exit'] = True
            position_data['max_loss_percent'] = trade_data.get('max_loss_percent', 5.0)
        
        monitor.add_position(trade_data['id'], position_data)
        
        if not monitor.running:
            monitor.start_monitoring()
