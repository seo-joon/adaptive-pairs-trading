from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from pairs_trading.config import load_config
from pairs_trading.engine import run_backtest

# KEY ARGUMENTS

# (just a few pairs to test with)
DEFAULT_PAIRS = [("XOM", "CVX"), ("GS", "MS"), ("V", "MA"), ("KO", "PEP"), ("HD", "LOW"), ("T", "VZ"), ("UPS", "FDX"), ("UNP", "CSV"), ("XLE", "OIH"), ("XLV", "IHI"), ("IWM", "SPY")]
DEFAULT_PAIRS = [
    # S&P500 — Ranked pairs (best → worst)
    ("KO", "PEP"),
    ("V", "MA"),
    ("XOM", "CVX"),
    ("HD", "LOW"),
    ("MS", "GS"),
    ("UPS", "FDX"),
    ("UNP", "CSX"),
    ("SLB", "HAL"),
    ("LMT", "NOC"),
    ("T", "VZ"),
    ("CAT", "DE"),
    ("PSA", "EXR"),
    ("MAR", "HLT"),
    ("TMO", "DHR"),
    ("JPM", "BAC"),
    ("AEP", "DUK"),

    # ETF pairs
    ("XLE", "OIH"),
    ("XLV", "IHI"),
    ("XLF", "KBE"),
    ("IYR", "VNQ"),
    ("IWM", "SPY"),
]
# DEFAULT_PAIRS = [("XOM", "CVX")]

FILE_NAME = "all-ohlcv-1m.csv"
BAR_MINUTES = 1
OUTPUT_DIR = "out-all-adaptive"
START_DATE = '2022-09-01'
END_DATE = '2023-09-01'


def make_synth(n_days: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    # Generate RTH minutes only
    idx = pd.date_range("2024-01-01 09:30", periods=n_days, freq="1D")
    rth_times = []
    for day in idx:
        if day.weekday() < 5:  # Mon-Fri only
            day_rth = pd.date_range(day.replace(hour=9, minute=30), day.replace(hour=16, minute=0), freq="1min")
            rth_times.extend(day_rth)
    idx = pd.DatetimeIndex(rth_times)

    x = 100 + np.cumsum(rng.normal(0, 0.05, size=len(idx)))
    beta = 1.2
    alpha = 0.5
    noise = rng.normal(0, 0.1, size=len(idx))
    y = alpha + beta * x + noise

    def to_df(sym: str, prices: np.ndarray):
        return pd.DataFrame({
            "symbol": sym,
            "datetime": idx,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1000,
        })

    df = pd.concat([to_df("X", x), to_df("Y", y)])
    return df.reset_index(drop=True)


def get_data(csv_file_path: str, pairs: list[tuple[str, str]]):
    # Create set of all symbols from pairs for efficient filtering
    required_symbols = set()
    for pair in pairs:
        required_symbols.add(pair[0])
        required_symbols.add(pair[1])
    
    df = pd.read_csv(csv_file_path)
    
    # Filter by required symbols early to reduce memory usage
    df = df[df['symbol'].isin(required_symbols)]

    # 1) standardize timestamp column name and type (DBN uses UTC ns)
    ts_col = 'ts' if 'ts' in df.columns else 'ts_event'
    df = df.rename(columns={ts_col: 'ts'}).copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True)

    # 2) set index and temporarily view in New York time
    df_idx = df.set_index('ts')
    df_ny = df_idx.tz_convert('America/New_York')

    # 3) keep only RTH (handles DST automatically)
    rth_ny = df_ny.between_time('09:30', '16:00')

    # 4) convert back to UTC for storage/consistency (optional)
    rth = rth_ny.tz_convert('UTC').reset_index()   # 'ts' is back in UTC

    # Map to required schema: rename ts->datetime
    rth = rth.rename(columns={"ts": "datetime"})


    # Filter by time
    start_date = pd.Timestamp(START_DATE, tz='UTC')
    end_date = pd.Timestamp(END_DATE, tz='UTC')
    rth = rth[(rth['datetime'] >= start_date) & (rth['datetime'] <= end_date)]

    # assume rth has columns: ['datetime','symbol','open','high','low','close','volume']
    rth = rth.sort_values(['symbol','datetime'])
    rth = (rth.set_index('datetime')
          .groupby('symbol')
          .resample(f'{BAR_MINUTES}min')
          .agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
          .dropna(subset=['open','high','low','close'])
          .reset_index())
    
    print('got data')

    return rth



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, default=None, help="Path to OHLCV CSV")
    args = parser.parse_args()
    

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    """# Use synthetic data for now
    df = make_synth(30)
    pairs = [("X", "Y")]
    cfg = load_config(None)"""

    # Use Real data
    data_path = args.data or os.path.join(os.path.dirname(__file__), "data", FILE_NAME)
    df = get_data(data_path, DEFAULT_PAIRS)
    
    cfg = load_config(None)
    cfg.screening.bar_minutes = BAR_MINUTES
    res = run_backtest(df, DEFAULT_PAIRS, cfg)

    res["ledger"].to_csv(os.path.join(OUTPUT_DIR, "ledger.csv"), index=False)
    res["equity"].to_csv(os.path.join(OUTPUT_DIR, "equity.csv"), header=["equity"])
    res["cash"].to_csv(os.path.join(OUTPUT_DIR, "cash.csv"), header=["cash"])

    # Minimal metrics json
    metrics = {
        "num_trades": int(len(res["ledger"])),
        "starting_cash": float(cfg.execution.starting_cash),
        "max_allocation_pct": float(cfg.execution.max_allocation_pct),
        "final_cash": float(res["cash"].iloc[-1]) if len(res["cash"]) > 0 else 0.0
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()


