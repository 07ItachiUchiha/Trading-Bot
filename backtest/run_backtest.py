import backtrader as bt
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import the strategy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.backtest_strategy import VolatilityBreakoutStrategy

def run_backtest(datafile='BTCUSDT_1h.csv', start_cash=10000.0, commission=0.001):
    """
    Run a backtest of the volatility breakout strategy
    
    Parameters:
    - datafile: CSV file with OHLCV data
    - start_cash: Initial capital
    - commission: Commission rate (e.g., 0.001 = 0.1%)
    
    Returns:
    - Results dictionary with performance metrics
    """
    # Initialize cerebro engine
    cerebro = bt.Cerebro()
    
    # Add our strategy
    cerebro.addstrategy(VolatilityBreakoutStrategy)
    
    # Check if file exists in data directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', datafile)
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found")
        return None
    
    # Load data
    data = bt.feeds.GenericCSVData(
        dataname=data_path,
        datetime=0,  # Column index for date
        open=1,      # Column index for open
        high=2,      # Column index for high
        low=3,       # Column index for low
        close=4,     # Column index for close
        volume=5,    # Column index for volume
        dtformat='%Y-%m-%d %H:%M:%S',
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # For hourly data
        openinterest=-1  # No open interest data
    )
    cerebro.adddata(data)
    
    # Set broker parameters
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)  # 0.1% commission
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Print starting cash
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Run strategy
    results = cerebro.run()
    strat = results[0]
    
    # Print results
    end_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {end_value:.2f}')
    
    # Performance metrics
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analysis = strat.analyzers.trades.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    
    print(f'Sharpe Ratio: {sharpe_ratio:.3f}')
    print(f'Max Drawdown: {drawdown.max.drawdown:.2f}%')
    
    # Trade statistics
    if trade_analysis.total.closed > 0:
        win_rate = 100 * trade_analysis.won.total / trade_analysis.total.closed
        avg_win = trade_analysis.won.pnl.average if trade_analysis.won.total > 0 else 0
        avg_loss = trade_analysis.lost.pnl.average if trade_analysis.lost.total > 0 else 0
        best_trade = trade_analysis.won.pnl.max if trade_analysis.won.total > 0 else 0
        worst_trade = trade_analysis.lost.pnl.max if trade_analysis.lost.total > 0 else 0
        
        print("\nTrade Statistics:")
        print(f"Total Trades: {trade_analysis.total.closed}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Best Trade: {best_trade:.2f}")
        print(f"Worst Trade: {worst_trade:.2f}")
    else:
        print("\nNo trades executed during the backtest period")
    
    # Plot the results
    cerebro.plot(style='candlestick', barup='green', bardown='red', volup='green', voldown='red',
                 plotdist=0.1, subplot=True, volume=True)
    
    # Return results dictionary
    results_dict = {
        'start_cash': start_cash,
        'end_value': end_value,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': drawdown.max.drawdown,
        'total_trades': trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0,
        'win_rate': win_rate if trade_analysis.total.closed > 0 else 0,
        'total_return': (end_value / start_cash - 1) * 100
    }
    
    return results_dict

def compare_parameters(params_list, datafile='BTCUSDT_1h.csv', start_cash=10000.0):
    """
    Run multiple backtests with different parameters for optimization
    
    Parameters:
    - params_list: List of dictionaries with parameters to test
    - datafile: CSV file with OHLCV data
    - start_cash: Initial capital
    
    Returns:
    - DataFrame with results for each parameter set
    """
    results = []
    
    for i, params in enumerate(params_list):
        print(f"\nRunning backtest {i+1}/{len(params_list)} with parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(VolatilityBreakoutStrategy, **params)
        
        # Load data
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', datafile)
        if not os.path.exists(data_path):
            print(f"Error: Data file {data_path} not found")
            continue
            
        data = bt.feeds.GenericCSVData(
            dataname=data_path,
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            dtformat='%Y-%m-%d %H:%M:%S',
            timeframe=bt.TimeFrame.Minutes,
            compression=60,
            openinterest=-1
        )
        cerebro.adddata(data)
        
        # Set broker parameters
        cerebro.broker.setcash(start_cash)
        cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run strategy
        strats = cerebro.run()
        strat = strats[0]
        
        # Get results
        end_value = cerebro.broker.getvalue()
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        drawdown = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        # Create results entry
        result = params.copy()
        result['end_value'] = end_value
        result['sharpe_ratio'] = sharpe_ratio
        result['max_drawdown'] = drawdown.max.drawdown if hasattr(drawdown.max, 'drawdown') else 0
        result['total_trades'] = trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0
        
        if hasattr(trade_analysis, 'total') and trade_analysis.total.closed > 0:
            result['win_rate'] = 100 * trade_analysis.won.total / trade_analysis.total.closed
        else:
            result['win_rate'] = 0
            
        result['total_return'] = (end_value / start_cash - 1) * 100
        
        results.append(result)
        
        print(f"Result: Return={result['total_return']:.2f}%, Sharpe={sharpe_ratio:.2f}, Win Rate={result['win_rate']:.1f}%")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by total return
    results_df = results_df.sort_values('total_return', ascending=False)
    
    return results_df

if __name__ == "__main__":
    # Run a simple backtest
    print("Running Volatility Breakout Strategy Backtest")
    run_backtest()
    
    # Example of parameter optimization
    """
    params_to_test = [
        # Test different BB squeeze thresholds
        {'bb_squeeze_threshold': 0.08, 'risk_percent': 1.5, 'atr_multiplier': 1.5},
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 1.5, 'atr_multiplier': 1.5},
        {'bb_squeeze_threshold': 0.12, 'risk_percent': 1.5, 'atr_multiplier': 1.5},
        
        # Test different ATR multipliers
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 1.5, 'atr_multiplier': 1.3},
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 1.5, 'atr_multiplier': 1.7},
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 1.5, 'atr_multiplier': 2.0},
        
        # Test different risk percentages
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 1.0, 'atr_multiplier': 1.5},
        {'bb_squeeze_threshold': 0.10, 'risk_percent': 2.0, 'atr_multiplier': 1.5},
    ]
    
    results = compare_parameters(params_to_test)
    print("\nParameter Optimization Results:")
    print(results.to_string())
    """