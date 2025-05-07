import backtrader as bt
import numpy as np
from datetime import datetime

class VolatilityBreakoutStrategy(bt.Strategy):
    """
    Backtrader implementation of the Volatility Breakout Strategy
    
    This strategy detects consolidation periods followed by volatility breakouts
    with volume confirmation and implements a comprehensive risk management system.
    """
    params = (
        ('bb_period', 20),
        ('rsi_period', 14),
        ('ema_fast', 50),
        ('ema_slow', 200),
        ('atr_period', 14),
        ('atr_multiplier', 1.5),
        ('bb_squeeze_threshold', 0.1),
        ('risk_percent', 1.5),
        ('rr_targets', [1.5, 2.5, 4.0]),
        ('consolidation_periods', 5),
    )
    
    def __init__(self):
        # Core indicators - using built-in Backtrader indicators instead of TA-Lib
        self.bbands = bt.ind.BollingerBands(period=self.p.bb_period)
        self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
        self.atr = bt.ind.ATR(period=self.p.atr_period)
        
        # Custom calculations
        self.bb_width = (self.bbands.lines.top - self.bbands.lines.bot) / self.bbands.lines.mid
        self.rsi_change = bt.ind.ChangeRate(self.rsi, periods=3)
        
        # Volume indicators
        self.volume_sma = bt.ind.SMA(self.data.volume, period=50)
        
        # Variables for tracking positions
        self.order = None
        self.stop_order = None
        self.target_orders = []
        self.entry_price = None
        self.stop_price = None
        self.targets = []
        self.breakout_signal = None
    
    def is_consolidation(self):
        # Check if BBand width is below threshold for multiple periods
        squeezed = True
        for i in range(1, self.p.consolidation_periods + 1):
            if self.bb_width[-i] > self.p.bb_squeeze_threshold:
                squeezed = False
                break
        
        # Check if ATR is declining
        atr_declining = self.atr[0] < self.atr[-5]
        
        return squeezed and atr_declining
    
    def next(self):
        # Skip if we have a pending order
        if self.order:
            return
        
        # Check if we're in a position
        if not self.position:
            # Check for consolidation
            consolidation = self.is_consolidation()
            
            # Check for bullish breakout
            if consolidation and self.data.close[0] > self.bbands.lines.top[0]:
                # Volume confirmation
                volume_surge = self.data.volume[0] > 1.5 * self.volume_sma[0]
                
                # Momentum confirmation
                rsi_momentum = self.rsi_change[0] > 5
                
                # Trend confirmation
                trend_bullish = self.ema_fast[0] > self.ema_slow[0]
                
                if volume_surge and rsi_momentum and trend_bullish:
                    # Calculate entry and stop
                    self.entry_price = self.data.close[0]
                    self.stop_price = self.entry_price - (self.atr[0] * self.p.atr_multiplier)
                    
                    # Calculate position size
                    risk_amount = self.broker.getvalue() * (self.p.risk_percent / 100)
                    sl_distance = self.entry_price - self.stop_price
                    size = risk_amount / sl_distance
                    
                    # Place market order
                    self.order = self.buy(size=size)
                    self.breakout_signal = 'long'
                    
                    # Calculate targets
                    self.targets = []
                    risk = self.entry_price - self.stop_price
                    for rr in self.p.rr_targets:
                        self.targets.append(self.entry_price + (risk * rr))
                    
                    # Place stop order
                    self.stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=self.stop_price)
                    
                    # Place partial take profit orders (33% at first target, half of remainder at second)
                    self.target_orders = []
                    tp1_size = size * 0.33
                    self.target_orders.append(self.sell(size=tp1_size, exectype=bt.Order.Limit, price=self.targets[0]))
                    
                    tp2_size = (size - tp1_size) * 0.5
                    self.target_orders.append(self.sell(size=tp2_size, exectype=bt.Order.Limit, price=self.targets[1]))
                    
                    # Rest will exit at last target or via trailing stop
                    tp3_size = size - tp1_size - tp2_size
                    self.target_orders.append(self.sell(size=tp3_size, exectype=bt.Order.Limit, price=self.targets[2]))
                    
                    self.log(f"LONG Entry: {self.entry_price:.2f}, Stop: {self.stop_price:.2f}, Targets: {[f'{t:.2f}' for t in self.targets]}")
            
            # Check for bearish breakout
            elif consolidation and self.data.close[0] < self.bbands.lines.bot[0]:
                # Volume confirmation
                volume_surge = self.data.volume[0] > 1.5 * self.volume_sma[0]
                
                # Momentum confirmation
                rsi_momentum = self.rsi_change[0] < -5
                
                # Trend confirmation
                trend_bearish = self.ema_fast[0] < self.ema_slow[0]
                
                if volume_surge and rsi_momentum and trend_bearish:
                    # Calculate entry and stop
                    self.entry_price = self.data.close[0]
                    self.stop_price = self.entry_price + (self.atr[0] * self.p.atr_multiplier)
                    
                    # Calculate position size
                    risk_amount = self.broker.getvalue() * (self.p.risk_percent / 100)
                    sl_distance = self.stop_price - self.entry_price
                    size = risk_amount / sl_distance
                    
                    # Place market order
                    self.order = self.sell(size=size)
                    self.breakout_signal = 'short'
                    
                    # Calculate targets
                    self.targets = []
                    risk = self.stop_price - self.entry_price
                    for rr in self.p.rr_targets:
                        self.targets.append(self.entry_price - (risk * rr))
                    
                    # Place stop order
                    self.stop_order = self.buy(size=size, exectype=bt.Order.Stop, price=self.stop_price)
                    
                    # Place partial take profit orders
                    self.target_orders = []
                    tp1_size = size * 0.33
                    self.target_orders.append(self.buy(size=tp1_size, exectype=bt.Order.Limit, price=self.targets[0]))
                    
                    tp2_size = (size - tp1_size) * 0.5
                    self.target_orders.append(self.buy(size=tp2_size, exectype=bt.Order.Limit, price=self.targets[1]))
                    
                    # Rest will exit at last target or via trailing stop
                    tp3_size = size - tp1_size - tp2_size
                    self.target_orders.append(self.buy(size=tp3_size, exectype=bt.Order.Limit, price=self.targets[2]))
                    
                    self.log(f"SHORT Entry: {self.entry_price:.2f}, Stop: {self.stop_price:.2f}, Targets: {[f'{t:.2f}' for t in self.targets]}")
        
        else:
            # Position management - update trailing stop
            if self.breakout_signal == 'long':
                price_movement = self.data.close[0] - self.entry_price
                profit_atr_multiple = price_movement / self.atr[0] if price_movement > 0 else 0
                
                # Calculate new stop once we've moved 1 ATR in profit
                if profit_atr_multiple > 1.0:
                    atr_stop = self.data.close[0] - (self.atr[0] * (2.0 - min(profit_atr_multiple * 0.25, 1.0)))
                    # Only move stop up, never down
                    if atr_stop > self.stop_price:
                        # Cancel previous stop order and create new one
                        if self.stop_order:
                            self.broker.cancel(self.stop_order)
                        
                        position_size = self.position.size
                        target_filled = sum(not order.alive() for order in self.target_orders)
                        remaining_size = position_size * (1 - 0.33 - (0.33 * 0.5 * min(target_filled, 1)))
                        
                        self.stop_price = atr_stop
                        self.stop_order = self.sell(size=remaining_size, exectype=bt.Order.Stop, price=self.stop_price)
                        self.log(f"Updated LONG stop to: {self.stop_price:.2f}")
            
            elif self.breakout_signal == 'short':
                price_movement = self.entry_price - self.data.close[0]
                profit_atr_multiple = price_movement / self.atr[0] if price_movement > 0 else 0
                
                # Calculate new stop once we've moved 1 ATR in profit
                if profit_atr_multiple > 1.0:
                    atr_stop = self.data.close[0] + (self.atr[0] * (2.0 - min(profit_atr_multiple * 0.25, 1.0)))
                    # Only move stop down, never up
                    if atr_stop < self.stop_price:
                        # Cancel previous stop order and create new one
                        if self.stop_order:
                            self.broker.cancel(self.stop_order)
                        
                        position_size = abs(self.position.size)
                        target_filled = sum(not order.alive() for order in self.target_orders)
                        remaining_size = position_size * (1 - 0.33 - (0.33 * 0.5 * min(target_filled, 1)))
                        
                        self.stop_price = atr_stop
                        self.stop_order = self.buy(size=remaining_size, exectype=bt.Order.Stop, price=self.stop_price)
                        self.log(f"Updated SHORT stop to: {self.stop_price:.2f}")
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.5f}')
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.5f}')
                
            # If market order is filled, set the entry price
            if order == self.order:
                self.order = None
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.order = None
    
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')