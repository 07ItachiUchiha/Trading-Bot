def calculate_position_size(capital, risk_percent, entry_price, stop_loss, max_capital_per_trade=0.15):
    """Figure out how many shares/units to buy based on risk and SL distance."""
    # dollar risk
    risk_amount = capital * (risk_percent / 100)
    
    # position size from SL distance
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        return 0

    # Position size based on risk
    position_size_risk = risk_amount / sl_distance
    
    # cap it so we don't put too much in one trade
    position_size_max = (capital * max_capital_per_trade) / entry_price
    
    # use the smaller one
    position_size = min(position_size_risk, position_size_max)
    
    return round(position_size, 5)

def calculate_trailing_stop(current_price, direction, atr, profit_ticks, initial_stop):
    """ATR-based trailing stop that tightens as profit grows."""
    if direction == 'long':
        atr_stop = current_price - (atr * (2.0 - min(profit_ticks * 0.25, 1.0)))
        new_stop = max(atr_stop, initial_stop)  # only moves up
    else:
        atr_stop = current_price + (atr * (2.0 - min(profit_ticks * 0.25, 1.0)))
        new_stop = min(atr_stop, initial_stop)  # only moves down
    
    return new_stop

def manage_open_position(position, current_price, current_atr):
    """Check trailing stops and partial TP targets for an open position."""
    direction = position['direction']
    entry_price = position['entry_price']
    current_stop = position['stop_loss']
    targets = position['targets']
    filled_targets = position['filled_targets']
    size = position['size']
    
    # how far are we in ATR terms
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
    
    # stop loss check
    if (direction == 'long' and current_price <= current_stop) or \
       (direction == 'short' and current_price >= current_stop):
        return 'exit', new_stop, size
    
    # Check TP targets
    action = None
    exit_size = 0
    
    for i, target in enumerate(targets):
        if i in filled_targets:
            continue
            
        if (direction == 'long' and current_price >= target) or \
           (direction == 'short' and current_price <= target):
            # figure out how much to close
            if i == len(targets) - 1:  # last target = close everything
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
