from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron


def _ar1_params(u: pd.Series):
    u = pd.Series(u).dropna()
    if len(u) < 3:
        return np.nan, np.nan
    y = u.iloc[1:].values
    x = u.iloc[:-1].values
    x = sm.add_constant(x)
    model = sm.OLS(y, x, missing="drop").fit()
    phi = model.params[1]
    # half-life in bars: ln(2)/-ln(phi) for |phi|<1
    if np.abs(phi) >= 1:
        return phi, np.inf
    half_life = np.log(2) / -np.log(np.abs(phi))
    return phi, float(half_life)


def _variance_ratio(u: pd.Series, m: int) -> float:
    u = pd.Series(u).dropna()
    if len(u) < m + 2:
        return np.nan
    ru = u.diff().dropna()
    var1 = np.var(ru, ddof=1)
    rmk = u.diff(m).dropna()
    varm = np.var(rmk, ddof=1) / m
    return float(varm / var1) if var1 > 0 else np.nan


def screen_pair(
    x,
    y,
    min_obs: int = 100,
    bar_minutes: int = 1,
    hl_min_minutes: int = 10,
    hl_max_minutes: int = 180,
    use_kpss: bool = False,
):
    x = pd.Series(x).astype(float).dropna()
    y = pd.Series(y).astype(float).dropna()
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).ffill()
    y = y.reindex(idx).ffill()
    if len(x) < min_obs:
        return {"cointegrated": False, "reason": "insufficient_obs", "n": len(x)}

    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    r2 = float(model.rsquared)
    u = y - (alpha + beta * x)

    # Unit root tests on residuals
    adf_res = adfuller(u.values, autolag="AIC")
    adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
    pp_p = float(PhillipsPerron(u.values).pvalue)
    kpss_p = None
    if use_kpss:
        try:
            kpss_p = float(kpss(u.values, regression="c")[1])
        except Exception:
            kpss_p = None

    phi, hl_bars = _ar1_params(u)
    half_life_minutes = float(hl_bars * bar_minutes) if np.isfinite(hl_bars) else np.inf

    vr2 = _variance_ratio(u, 2)
    vr4 = _variance_ratio(u, 4)
    vr8 = _variance_ratio(u, 8)
    vr_ok = all([np.isnan(v) or v <= 1.0 for v in [vr2, vr4, vr8]])

    stationarity_pass = (adf_p < 0.05) and (pp_p < 0.05)
    if use_kpss and (kpss_p is not None):
        stationarity_pass = stationarity_pass and (kpss_p > 0.05)

    mean_reversion_pass = np.isfinite(hl_bars) and (hl_min_minutes <= half_life_minutes <= hl_max_minutes)
    decision = stationarity_pass and mean_reversion_pass and vr_ok

    return {
        "cointegrated": bool(decision),
        "alpha": alpha,
        "beta": beta,
        "r_squared": r2,
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "pp_p": pp_p,
        "kpss_p": kpss_p,
        "phi": float(phi) if np.isfinite(phi) else np.nan,
        "half_life_minutes": half_life_minutes,
        "vr2": vr2,
        "vr4": vr4,
        "vr8": vr8,
        "vr_ok": bool(vr_ok),
        "stationarity_pass": bool(stationarity_pass),
        "mean_reversion_pass": bool(mean_reversion_pass),
        "half_life_acceptable": bool(mean_reversion_pass),
        "reason": "ok" if decision else "fail",
        "n": int(len(x)),
    }


def screen_pairs(pairs: List[Tuple[str, str]], price_map: Dict[str, pd.Series], config) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        xa = price_map.get(a)
        yb = price_map.get(b)
        if xa is None or yb is None:
            rows.append({"pair": f"{a}-{b}", "cointegrated": False, "reason": "missing_symbol"})
            continue
        idx = xa.index.intersection(yb.index)
        window_start = pd.Timestamp(idx.min()) if len(idx) else None
        window_end = pd.Timestamp(idx.max()) if len(idx) else None
        res = screen_pair(
            xa,
            yb,
            min_obs=config.screening.min_obs,
            bar_minutes=config.screening.bar_minutes,
            hl_min_minutes=config.screening.hl_min_minutes,
            hl_max_minutes=config.screening.hl_max_minutes,
            use_kpss=config.screening.use_kpss,
        )
        res.update({
            "pair": f"{a}-{b}",
            "symbol_y": b,
            "symbol_x": a,
            "window_start": window_start,
            "window_end": window_end,
        })
        if res.get("cointegrated"):
            try:
                print(
                    f"[{window_start}→{window_end}] Screen pass {a}-{b}: adf_p={res.get('adf_p'):.4f}, pp_p={res.get('pp_p'):.4f}, "
                    f"hl_min={res.get('half_life_minutes'):.1f}m, vr_ok={res.get('vr_ok')}"
                )
            except Exception:
                print(f"[{window_start}→{window_end}] Screen pass {a}-{b}")
        else:
            # Print key diagnostics for failed screens too
            adf_p = res.get('adf_p'); pp_p = res.get('pp_p'); hl = res.get('half_life_minutes')
            reason = res.get('reason', 'fail')
            try:
                print(f"[{window_start}→{window_end}] Screen fail {a}-{b}: reason={reason}, adf_p={adf_p:.4f}, pp_p={pp_p:.4f}, hl_min={hl:.1f}m")
            except Exception:
                print(f"[{window_start}→{window_end}] Screen fail {a}-{b}: reason={reason}")
        rows.append(res)
    return pd.DataFrame(rows)


