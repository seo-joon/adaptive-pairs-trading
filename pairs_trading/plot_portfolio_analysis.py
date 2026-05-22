# analysis_portfolio_metrics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import load_config

config = load_config(None)
train_lookback_days = config.engine.train_lookback_days
starting_cash = config.execution.starting_cash

# Read equity data
df_equity = pd.read_csv('out-all-adaptive/equity.csv', index_col=0, parse_dates=True)
df_equity = df_equity.sort_index().copy()
df_equity['equity'] = pd.to_numeric(df_equity['equity'], errors='coerce')
df_equity = df_equity.dropna(subset=['equity'])

if df_equity.empty:
    raise ValueError("Equity dataframe is empty after cleaning.")

# Convert to portfolio value
df_equity['portfolio_value'] = starting_cash + df_equity['equity']

# Exclude training period
trading_start_date = df_equity.index[0] + pd.Timedelta(days=train_lookback_days)
df_trading = df_equity[df_equity.index >= trading_start_date].copy()

if df_trading.empty:
    print(f"Warning: No trading data after {train_lookback_days} days training.")
    df_trading = df_equity.copy()

# ============================================
# FIXED: Use DAILY returns for Sharpe
# ============================================
df_daily = df_trading['portfolio_value'].resample('1D').last().dropna()
daily_returns = df_daily.pct_change().dropna()


# Risk-free rate
rf_annual = 0.05
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Excess returns
excess_daily = daily_returns - rf_daily

# Sharpe ratio
mu_daily = excess_daily.mean()
sigma_daily = excess_daily.std(ddof=1)

if sigma_daily > 0:
    sharpe_annual = np.sqrt(252) * (mu_daily / sigma_daily)
else:
    sharpe_annual = np.nan

# Sortino ratio
downside_daily = excess_daily[excess_daily < 0]
dd_daily = downside_daily.std(ddof=1) if len(downside_daily) > 0 else np.nan

if dd_daily > 0 and np.isfinite(dd_daily):
    sortino_annual = np.sqrt(252) * (mu_daily / dd_daily)
else:
    sortino_annual = np.nan

# Additional metrics
total_return = (df_daily.iloc[-1] / df_daily.iloc[0]) - 1
trading_days = len(df_daily)
years_elapsed = trading_days / 252.0
cagr = ((1 + total_return) ** (1 / years_elapsed) - 1) if years_elapsed > 0 else np.nan
ann_vol = sigma_daily * np.sqrt(252)

# Maximum drawdown
cumulative = df_daily / df_daily.iloc[0]
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()

# Plotting
plt.figure(figsize=(14, 10))

# Plot 1: Portfolio value
plt.subplot(3, 1, 1)
plt.plot(df_daily.index, df_daily.values, linewidth=1.5, alpha=0.85)
plt.title('Portfolio Value Over Time (Daily)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Daily returns
plt.subplot(3, 1, 2)
plt.hist(daily_returns, bins=50, alpha=0.8, edgecolor='black')
plt.title(f'Daily Returns Distribution (n={len(daily_returns)})', fontsize=14, fontweight='bold')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot 3: Drawdown
plt.subplot(3, 1, 3)
plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red')
plt.plot(drawdown.index, drawdown.values * 100, linewidth=1, color='darkred')
plt.title('Drawdown (%)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()

# Print statistics
print("=" * 70)
print("PORTFOLIO ANALYSIS RESULTS (DAILY RETURNS)")
print("=" * 70)
print(f"Training period excluded:     {train_lookback_days} days")
print(f"Trading start date:           {trading_start_date.date()}")
print(f"Starting Cash:                ${starting_cash:,.2f}")
print(f"Final Portfolio Value:        ${df_daily.iloc[-1]:,.2f}")
print(f"Total Return:                 {total_return*100:,.2f}%")
print(f"Total P&L:                    ${df_daily.iloc[-1] - starting_cash:,.2f}")
print()

print("RISK METRICS (ANNUALIZED FROM DAILY RETURNS)")
print("-" * 70)
print(f"Risk-free rate (annual):      {rf_annual:.2%}")
print(f"Sharpe Ratio:                 {sharpe_annual:,.4f}")
print(f"Sortino Ratio:                {sortino_annual:,.4f}")
print(f"CAGR:                         {cagr*100:,.2f}%")
print(f"Annualized Volatility:        {ann_vol*100:,.2f}%")
print(f"Max Drawdown:                 {max_dd*100:,.2f}%")
print(f"Max DD Date:                  {max_dd_date.date()}")
print()

print("RETURN STATISTICS")
print("-" * 70)
print(f"Mean Daily Return:            {daily_returns.mean()*100:,.4f}%")
print(f"Std Daily Return:             {daily_returns.std(ddof=1)*100:,.4f}%")
print(f"Mean Excess Daily Return:     {mu_daily*100:,.4f}%")
print(f"Trading Days:                 {trading_days:,d}")
print(f"Years Elapsed:                {years_elapsed:,.2f}")
print(f"Winning Days:                 {(daily_returns > 0).sum():,d} ({(daily_returns > 0).sum()/len(daily_returns)*100:.1f}%)")
print(f"Losing Days:                  {(daily_returns < 0).sum():,d} ({(daily_returns < 0).sum()/len(daily_returns)*100:.1f}%)")

plt.show()