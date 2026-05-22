import numpy as np
import pandas as pd

def kelly_from_spread(spread: pd.Series) -> float:
    du = spread.diff().dropna()
    if du.empty:
        return 0.0
    expected_ret = -np.nanmean(np.sign(spread.shift(1).reindex(du.index)) * du)
    vol = np.nanstd(du, ddof=0)
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    f = expected_ret / (vol ** 2)
    return max(0.0, float(f))

def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))
