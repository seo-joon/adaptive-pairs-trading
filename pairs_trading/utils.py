from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype


REQUIRED_COLUMNS = ["symbol", "datetime", "open", "high", "low", "close", "volume"]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not is_datetime64_any_dtype(df["datetime"]) and not is_datetime64tz_dtype(df["datetime"]):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df


def check_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def to_bps(x: float) -> float:
    return x / 10_000.0


