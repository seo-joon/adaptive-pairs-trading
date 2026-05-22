# cash_management_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import load_config

# -----------------------------
# Config & data
# -----------------------------
config = load_config(None)
train_lookback_days = config.engine.train_lookback_days
starting_cash = config.execution.starting_cash
max_allocation_pct = config.execution.max_allocation_pct

# Read the cash data (index is timestamps, column 'cash')
df_cash = pd.read_csv('out-all-adaptive/cash.csv', index_col=0, parse_dates=True)
df_cash = df_cash.sort_index().copy()
df_cash['cash'] = pd.to_numeric(df_cash['cash'], errors='coerce')
df_cash = df_cash.dropna(subset=['cash'])

if df_cash.empty:
    raise ValueError("Cash dataframe is empty after cleaning. Check input file.")

# Exclude training period from analysis
trading_start_date = df_cash.index[0] + pd.Timedelta(days=train_lookback_days)
df_trading = df_cash[df_cash.index >= trading_start_date].copy()
if df_trading.empty:
    print(f"Warning: No trading data found after {train_lookback_days} days of training period. Using full data instead.")
    df_trading = df_cash.copy()

# -----------------------------
# Cash Management Metrics
# -----------------------------

# Cash utilization (how much of starting cash is tied up in positions)
df_trading['cash_utilization'] = (starting_cash - df_trading['cash']) / starting_cash
df_trading['available_cash_pct'] = df_trading['cash'] / starting_cash

# Cash drawdowns (from starting cash)
df_trading['cash_drawdown'] = (df_trading['cash'] - starting_cash) / starting_cash

# Rolling statistics for cash management
window_days = 5  # 5-day rolling window
df_trading['cash_5d_mean'] = df_trading['cash'].rolling(window=f'{window_days}D').mean()
df_trading['cash_5d_std'] = df_trading['cash'].rolling(window=f'{window_days}D').std()
df_trading['cash_5d_min'] = df_trading['cash'].rolling(window=f'{window_days}D').min()
df_trading['cash_5d_max'] = df_trading['cash'].rolling(window=f'{window_days}D').max()

# Identify periods of low cash (potential liquidity issues)
low_cash_threshold = starting_cash * 0.1  # 10% of starting cash
df_trading['low_cash_flag'] = df_trading['cash'] < low_cash_threshold

# Cash volatility (how much cash balance fluctuates)
cash_volatility = df_trading['cash'].std()

# Maximum cash utilization
max_utilization = df_trading['cash_utilization'].max()
min_available_cash = df_trading['available_cash_pct'].min()

# Time spent at different cash levels
high_cash_pct = (df_trading['available_cash_pct'] > 0.8).sum() / len(df_trading) * 100
medium_cash_pct = ((df_trading['available_cash_pct'] >= 0.5) & (df_trading['available_cash_pct'] <= 0.8)).sum() / len(df_trading) * 100
low_cash_pct = (df_trading['available_cash_pct'] < 0.5).sum() / len(df_trading) * 100

# Cash efficiency metrics
avg_cash_utilization = df_trading['cash_utilization'].mean()
cash_efficiency = 1 - (df_trading['cash'].mean() / starting_cash)  # Higher = more capital deployed

# -----------------------------
# Plots
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Cash balance over time with utilization
ax1 = axes[0]
ax1.plot(df_trading.index, df_trading['cash'], linewidth=1, alpha=0.8, label='Available Cash', color='blue')
ax1.axhline(y=starting_cash, color='green', linestyle='--', alpha=0.7, label='Starting Cash')
ax1.axhline(y=low_cash_threshold, color='red', linestyle='--', alpha=0.7, label='Low Cash Threshold (10%)')
ax1.fill_between(df_trading.index, df_trading['cash'], starting_cash, 
                where=(df_trading['cash'] < starting_cash), alpha=0.3, color='orange', label='Capital Deployed')
ax1.set_title('Cash Balance Over Time (Trading Period Only)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cash ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Cash utilization percentage
ax2 = axes[1]
ax2.plot(df_trading.index, df_trading['cash_utilization'] * 100, linewidth=1, alpha=0.8, color='purple')
ax2.axhline(y=max_allocation_pct * 100, color='red', linestyle='--', alpha=0.7, 
           label=f'Max Allocation Limit ({max_allocation_pct*100:.1f}%)')
ax2.set_title('Capital Utilization Over Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('Utilization (%)')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Available cash percentage with rolling statistics
ax3 = axes[2]
ax3.plot(df_trading.index, df_trading['available_cash_pct'] * 100, linewidth=1, alpha=0.6, color='blue', label='Available Cash %')
ax3.plot(df_trading.index, df_trading['cash_5d_mean'] / starting_cash * 100, linewidth=2, alpha=0.8, color='red', label='5-Day Rolling Mean')
ax3.fill_between(df_trading.index, 
                (df_trading['cash_5d_mean'] - df_trading['cash_5d_std']) / starting_cash * 100,
                (df_trading['cash_5d_mean'] + df_trading['cash_5d_std']) / starting_cash * 100,
                alpha=0.2, color='red', label='5-Day Rolling ±1σ')
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Low Cash Threshold (10%)')
ax3.set_title('Available Cash Percentage with Rolling Statistics', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Available Cash (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()

# -----------------------------
# Print Cash Management Statistics
# -----------------------------
print("=" * 70)
print("CASH MANAGEMENT ANALYSIS (TRADING PERIOD ONLY)")
print("=" * 70)
print(f"Training period excluded:     {train_lookback_days} days (from config)")
print(f"Trading start date:          {trading_start_date}")
print(f"Starting Cash:               ${starting_cash:,.2f}")
print(f"Max Allocation per Trade:    {max_allocation_pct*100:.1f}% (${starting_cash * max_allocation_pct:,.2f})")
print()

print("CASH LEVELS & UTILIZATION")
print("-" * 70)
print(f"Current Cash:                ${df_trading['cash'].iloc[-1]:,.2f}")
print(f"Max Cash:                    ${df_trading['cash'].max():,.2f}")
print(f"Min Cash:                    ${df_trading['cash'].min():,.2f}")
print(f"Mean Cash:                   ${df_trading['cash'].mean():,.2f}")
print(f"Cash Volatility (Std):       ${cash_volatility:,.2f}")
print()

print("CAPITAL DEPLOYMENT")
print("-" * 70)
print(f"Max Capital Utilization:     {max_utilization*100:.2f}% (${starting_cash * max_utilization:,.2f})")
print(f"Min Available Cash:          {min_available_cash*100:.2f}% (${starting_cash * min_available_cash:,.2f})")
print(f"Avg Capital Utilization:     {avg_cash_utilization*100:.2f}%")
print(f"Cash Efficiency:             {cash_efficiency*100:.2f}% (Higher = more capital deployed)")
print()

print("CASH LEVEL DISTRIBUTION")
print("-" * 70)
print(f"High Cash (>80%):            {high_cash_pct:.1f}% of time")
print(f"Medium Cash (50-80%):        {medium_cash_pct:.1f}% of time")
print(f"Low Cash (<50%):             {low_cash_pct:.1f}% of time")
print()

print("LIQUIDITY ANALYSIS")
print("-" * 70)
low_cash_periods = df_trading['low_cash_flag'].sum()
total_periods = len(df_trading)
print(f"Low Cash Periods:            {low_cash_periods:,d} out of {total_periods:,d} ({low_cash_periods/total_periods*100:.2f}%)")
if low_cash_periods > 0:
    low_cash_days = df_trading[df_trading['low_cash_flag']].index.normalize().nunique()
    print(f"Low Cash Days:               {low_cash_days:,d} unique days")
    print(f"Lowest Cash Level:           ${df_trading['cash'].min():,.2f} ({df_trading['cash'].min()/starting_cash*100:.2f}%)")

print()
print("RISK ASSESSMENT")
print("-" * 70)
if min_available_cash < 0.1:
    print("⚠️  WARNING: Cash dropped below 10% of starting capital")
elif min_available_cash < 0.2:
    print("⚠️  CAUTION: Cash dropped below 20% of starting capital")
else:
    print("✅ Cash levels maintained above 20% of starting capital")

if max_utilization > 0.9:
    print("⚠️  WARNING: Capital utilization exceeded 90%")
elif max_utilization > 0.8:
    print("⚠️  CAUTION: Capital utilization exceeded 80%")
else:
    print("✅ Capital utilization kept below 80%")

plt.show()