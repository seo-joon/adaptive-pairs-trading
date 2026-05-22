from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import REQUIRED_COLUMNS, ensure_datetime_index, check_required_columns


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df, REQUIRED_COLUMNS)
    df = ensure_datetime_index(df)
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    # Basic 1-minute frequency check per symbol (tolerant to missing bars)
    # Raise if we find sub-minute gaps
    grp = df.groupby("symbol")
    for sym, g in grp:
        if len(g) < 2:
            continue
        diffs = g["datetime"].diff().dropna().dt.total_seconds()
        if (diffs < 60).any():
            raise ValueError(f"Sub-minute bars detected for symbol {sym}")
    return df


def default_us_rth(ts: pd.Timestamp) -> bool:
    ts_local = ts.tz_convert("America/New_York") if ts.tzinfo else ts.tz_localize("America/New_York")
    is_weekday = ts_local.weekday() < 5
    open_time = ts_local.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = ts_local.replace(hour=16, minute=0, second=0, microsecond=0)
    return bool(is_weekday and (ts_local >= open_time) and (ts_local < close_time))


def get_rth_mask(df: pd.DataFrame, calendar: Optional[Callable[[pd.Timestamp], bool]] = None) -> pd.Series:
    if calendar is None:
        calendar = default_us_rth
    mask = df["datetime"].apply(calendar)
    return mask


def slice_df(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["datetime"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["datetime"] <= pd.Timestamp(end)
    return df.loc[m].copy()


def window_df(df: pd.DataFrame, end: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=lookback_days)
    return slice_df(df, start, end)


def wide_prices(
    df: pd.DataFrame,
    price: str = "close",
    rth_only: bool = True,
    calendar: Optional[Callable[[pd.Timestamp], bool]] = None,
) -> Dict[str, pd.Series]:
    df = validate_dataframe(df)
    if rth_only:
        mask = get_rth_mask(df, calendar)
        df = df.loc[mask]

    out: Dict[str, pd.Series] = {}
    for sym, g in df.groupby("symbol"):
        s = g.set_index("datetime")[price].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        out[sym] = s

    # Align indices across symbols
    all_index = None
    for s in out.values():
        all_index = s.index if all_index is None else all_index.union(s.index)
    if all_index is None:
        return {}
    for sym in list(out.keys()):
        out[sym] = out[sym].reindex(all_index).ffill()
    return out


