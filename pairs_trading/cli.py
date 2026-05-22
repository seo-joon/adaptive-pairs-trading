from __future__ import annotations

import argparse
import os

import pandas as pd

from .config import load_config
from .data import validate_dataframe
from .engine import run_backtest
from .plots import plot_equity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--config", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    df = validate_dataframe(df)

    pairs_df = pd.read_csv(args.pairs)
    pairs = list(map(tuple, pairs_df[["a", "b"]].values))

    results = run_backtest(df, pairs, cfg)

    results["ledger"].to_csv(os.path.join(args.out, "ledger.csv"), index=False)
    results["equity"].to_csv(os.path.join(args.out, "equity.csv"), header=["equity"]) 
    plot_equity(results["equity"], os.path.join(args.out, "equity.png"))


if __name__ == "__main__":
    main()


