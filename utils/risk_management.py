def calculate_position_size(capital, risk_percent, entry_price, stop_loss, max_capital_per_trade=0.15):
    """
    Calculate position size based on risk percentage and ATR
    
    Parameters:
    - capital: Total trading capital
    - risk_percent: Max percentage of capital to risk (e.g., 1.5 for 1.5%)
    - entry_price: Entry price of the trade
    - stop_loss: Stop loss price
    - max_capital_per_trade: Maximum percentage of capital to allocate to a single trade
    
    Returns: Position size
    """
    # Calculate dollar risk amount
    risk_amount = capital * (risk_percent / 100)
    
    # Calculate position size based on risk
    sl_distance = abs(entry_price - stop_loss)
    sl_percentage = sl_distance / entry_price
    
    # Position size based on risk
    position_size_risk = risk_amount / sl_distance
    
    # Position size based on max capital allocation
    position_size_max = (capital * max_capital_per_trade) / entry_price
    
    # Use the smaller of the two position sizes
    position_size = min(position_size_risk, position_size_max)
    
    return round(position_size, 5)

def calculate_trailing_stop(current_price, direction, atr, profit_ticks, initial_stop):
    """
    Calculate a trailing stop that increases as the trade moves in your favor
    
    Parameters:
    - current_price: Current price
    - direction: 'long' or 'short'
    - atr: Current ATR value
    - profit_ticks: How many ATRs price has moved in your favor 
    - initial_stop: The initial stop loss
    
    Returns: New stop loss price
    """
    if direction == 'long':
        # For long positions, calculate trailing stop below current price
        atr_stop = current_price - (atr * (2.0 - min(profit_ticks * 0.25, 1.0)))
        # Only move stop up, never down
        new_stop = max(atr_stop, initial_stop)
    else:
        # For short positions, calculate trailing stop above current price
        atr_stop = current_price + (atr * (2.0 - min(profit_ticks * 0.25, 1.0)))
        # Only move stop down, never up
        new_stop = min(atr_stop, initial_stop)
    
    return new_stop

def manage_open_position(position, current_price, current_atr):
    """
    Manage an open position with trailing stops and partial take-profits
    
    Parameters:
    - position: Dictionary containing position details
    - current_price: Current market price
    - current_atr: Current ATR value
    
    Returns:
    - action: None, 'partial_exit', or 'exit'
    - new_stop: Updated stop loss price
    - exit_size: Size to exit (if applicable)
    """
    direction = position['direction']
    entry_price = position['entry_price']
    current_stop = position['stop_loss']
    targets = position['targets']
    filled_targets = position['filled_targets']
    size = position['size']
    
    # Calculate how many ATRs we've moved in profit
    if direction == 'long':
        price_movement = current_price - entry_price
        is_profit = current_price > entry_price
    else:  # short
        price_movement = entry_price - current_price
        is_profit = current_price < entry_price
    
    profit_atr_multiple = price_movement / current_atr if is_profit else 0
    
    # Update trailing stop
    new_stop = calculate_trailing_stop(
        current_price, direction, current_atr, profit_atr_multiple, current_stop
    )
    
    # Check if price hit stop loss
    if (direction == 'long' and current_price <= current_stop) or \
       (direction == 'short' and current_price >= current_stop):
        return 'exit', new_stop, size
    
    # Check for take-profit targets
    action = None
    exit_size = 0
    
    for i, target in enumerate(targets):
        if i in filled_targets:
            continue
            
        if (direction == 'long' and current_price >= target) or \
           (direction == 'short' and current_price <= target):
            # Determine size to exit at this target
            if i == len(targets) - 1:  # Last target - exit all remaining
                exit_size = size - sum(position.get('exit_sizes', [0]))
            else:
                # Exit 1/3 at first target, 1/2 of remainder at second
                if i == 0:
                    exit_size = size * 0.33
                else:
                    remaining = size - sum(position.get('exit_sizes', [0]))
                    exit_size = remaining * 0.5
            
            filled_targets.append(i)
            
            if 'exit_sizes' not in position:
                position['exit_sizes'] = []
            
            position['exit_sizes'].append(exit_size)
            
            if sum(position['exit_sizes']) >= size * 0.99:
                return 'exit', new_stop, exit_size
            else:
                return 'partial_exit', new_stop, exit_size
    
    return None, new_stop, 0