from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, scale: int = 252 * 6.5 * 60) -> float:
    mu = returns.mean() * np.sqrt(scale)
    sd = returns.std(ddof=1)
    return float(mu / sd) if sd > 0 else np.nan


def max_drawdown(series: pd.Series) -> Dict[str, float]:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    mdd = dd.min()
    end = dd.idxmin() if len(dd) else None
    start = series.loc[:end].idxmax() if end is not None else None
    duration = int((series.index.get_loc(end) - series.index.get_loc(start))) if (start is not None and end is not None) else 0
    return {"max_dd": float(mdd), "start": start, "end": end, "duration_bars": duration}


