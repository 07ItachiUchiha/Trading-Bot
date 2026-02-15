import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class RiskManager:
    """Handles position sizing, correlation checks, and portfolio-level risk limits."""
    
    def __init__(self, config=None):
        # defaults
        self.config = {
            'max_daily_loss': 3.0,
            'max_trade_loss': 1.0,
            'max_correlated_exposure': 15.0,
            'max_position_size': 0.1,
            'sector_limits': {
                'technology': 0.4,
                'finance': 0.3,
                'crypto': 0.2,
                'other': 0.5
            },
            'correlation_threshold': 0.7
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # portfolio tracking
        self.daily_pnl = 0
        self.positions = {}
        self.correlations = {}
        self.sector_exposure = {
            'technology': 0,
            'finance': 0,
            'crypto': 0,
            'other': 0
        }
        
        # sector mappings
        self.sector_map = {
            'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
            'AMZN': 'technology', 'META': 'technology', 'NFLX': 'technology',
            'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'GS': 'finance',
            'BTCUSD': 'crypto', 'ETHUSD': 'crypto', 'LTCUSD': 'crypto'
        }
        
        self.logger = logging.getLogger('RiskManager')
        
    def calculate_position_size(self, symbol, entry_price, stop_loss, confidence=0.7, account_value=None):
        """Calculate position size based on stop distance and risk limits."""
        # fallback if missing inputs
        if entry_price is None or stop_loss is None or entry_price == stop_loss:
            return self.config['max_position_size'] * confidence
        
        # Calculate risk amount (% of portfolio to risk)
        risk_percent = self.config['max_trade_loss'] * confidence
        risk_amount = (account_value or 100000) * (risk_percent / 100)
        
        # Calculate position size based on stop distance
        stop_distance_percent = abs(entry_price - stop_loss) / entry_price
        position_value = risk_amount / stop_distance_percent
        
        # Convert to position size as fraction of portfolio
        position_size = position_value / (account_value or 100000)
        
        # cap it
        position_size = min(position_size, self.config['max_position_size'])
        
        # sector limits
        sector = self.get_symbol_sector(symbol)
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        sector_limit = self.config['sector_limits'].get(sector, 0.5)
        
        if current_sector_exposure + position_size > sector_limit:
            position_size = max(0, sector_limit - current_sector_exposure)
            self.logger.info(f"Position size reduced due to sector limit for {sector}")
        
        # Apply correlation-based limits
        position_size = self.adjust_for_correlation(symbol, position_size)
        
        return position_size
    
    def adjust_for_correlation(self, symbol, position_size):
        """Reduce position size if too correlated with existing positions."""
        # If no correlation data or no positions, return original size
        if not self.correlations or not self.positions:
            return position_size
        
        # Calculate total correlated exposure
        correlated_exposure = 0
        for pos_symbol, pos_data in self.positions.items():
            if pos_symbol != symbol:
                correlation = self.correlations.get((symbol, pos_symbol), 0)
                
                # Only consider meaningful correlations
                if abs(correlation) >= self.config['correlation_threshold']:
                    correlated_exposure += pos_data['size'] * correlation
        
        # If correlated exposure is high, reduce position size
        max_correlation_exposure = self.config['max_correlated_exposure'] / 100
        if correlated_exposure > max_correlation_exposure:
            # Reduction factor
            reduction = 1 - ((correlated_exposure - max_correlation_exposure) / max_correlation_exposure)
            adjusted_size = position_size * max(0.1, reduction)  # Never reduce by more than 90%
            self.logger.info(f"Position size reduced due to correlation: {position_size:.4f} -> {adjusted_size:.4f}")
            return adjusted_size
            
        return position_size
    
    def get_symbol_sector(self, symbol):
        """Look up which sector a symbol belongs to."""
        return self.sector_map.get(symbol, 'other')
    
    def update_position(self, symbol, size, direction='long'):
        """Track a new/updated position."""
        sector = self.get_symbol_sector(symbol)
        
        # Track position
        self.positions[symbol] = {
            'size': size,
            'direction': direction,
            'sector': sector
        }
        
        # Update sector exposure
        self.sector_exposure[sector] = sum(
            pos['size'] for pos in self.positions.values() 
            if pos['sector'] == sector
        )
    
    def remove_position(self, symbol):
        """Remove a position from tracking."""
        if symbol in self.positions:
            sector = self.positions[symbol]['sector']
            size = self.positions[symbol]['size']
            
            # Update sector exposure
            self.sector_exposure[sector] -= size
            if self.sector_exposure[sector] < 0:
                self.sector_exposure[sector] = 0
                
            # Remove from positions
            del self.positions[symbol]
    
    def update_daily_pnl(self, pnl_change):
        """Track daily PnL. Returns True if we've hit the loss limit."""
        self.daily_pnl += pnl_change
        
        # Check if daily loss limit is exceeded
        if self.daily_pnl < -self.config['max_daily_loss']:
            self.logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}%")
            return True
            
        return False
    
    def reset_daily_tracking(self):
        """Reset daily PnL tracking"""
        self.daily_pnl = 0
    
    def update_correlation_matrix(self, price_data):
        """Recalculate correlation matrix from price data."""
        if not price_data:
            return
            
        # Create a unified DataFrame of returns
        returns_dict = {}
        for symbol, df in price_data.items():
            if 'close' in df.columns:
                returns_dict[symbol] = df['close'].pct_change().dropna()
        
        # Create a DataFrame from the dict
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Convert to dictionary for easier lookup
            for symbol1 in corr_matrix.index:
                for symbol2 in corr_matrix.columns:
                    self.correlations[(symbol1, symbol2)] = corr_matrix.loc[symbol1, symbol2]
