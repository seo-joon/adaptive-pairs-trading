# Adaptive Pairs Trading Engine
By Alex Zhang, Yusuf Zahran

## Structure
run_backtest.py # main function used to generate results  
pairs_trading/  
├─ config.py # parameters & configs  
├─ data.py # data loading & filtering  
├─ screen.py # cointegration screening  
├─ signals.py # entry/exit logic  
├─ engine.py # backtest engine  
├─ position.py # hedge ratios & exposure  
├─ broker.py # transaction costs  
├─ plots.py # equity curve plotting  
├─ utils.py  
├─ callbacks.py  
├─ metrics.py  
├─ plot_portfolio_analysis.py # plot portfolio performance over specified timeframe  
├─ plot_cash_analysis.py # plot cash analysis over specified timeframe  

## Instructions
Download and drop all-ohlcv-1m.csv into data/  
Run run_backtest.py to generate results in out-all-adaptive/  
Plot graphs using plot_portfolio_analysis.py, plot_cash_analysis.py  
