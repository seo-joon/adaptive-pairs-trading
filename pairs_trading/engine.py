from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import wide_prices, get_rth_mask
from .screen import screen_pairs
from .signals import entry_signal, exit_reason, AdaptiveZScore
from .position import apply_vol_scale
from .broker import simulate_fill
from .kelly import kelly_from_spread, clamp


@dataclass
class Trade:
    pair: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp | None
    side: int
    alpha: float
    beta: float
    qty_y: float
    qty_x: float
    entry_z: float
    exit_z: float | None
    pnl: float
    fees: float
    reason: str | None


def _weekly_anchors(index: pd.DatetimeIndex, train_lookback_days: int = 60, trade_forward_days: int = 365) -> List[pd.Timestamp]:
    """
    Create training/trading anchors based on trade_forward_days parameter.
    Starts with first Monday, then rolls forward by trade_forward_days each time.
    If trade_forward_days is large relative to dataset, only creates one anchor.
    """
    anchors = []
    by_day = pd.Series(1, index=index).resample("1D").sum()
    
    # Find first valid anchor (first Monday with enough training data)
    first_valid_anchor = None
    for day in by_day.index:
        if day.weekday() == 0:  # Monday
            day_idx = index[(index.date == day.date())]
            if len(day_idx) == 0:
                continue
            
            train_end = day_idx[0] - pd.Timedelta(minutes=1)
            train_start = train_end - pd.Timedelta(days=train_lookback_days)
            
            if train_start >= index[0]:
                first_valid_anchor = day_idx[0]
                break
    
    if first_valid_anchor is None:
        print(f"No valid anchors found - need at least {train_lookback_days} days of data before first Monday")
        return []
    
    # Calculate total dataset span in days
    dataset_start = index[0]
    dataset_end = index[-1]
    total_days = (dataset_end - dataset_start).days
    
    # If trade_forward_days is large relative to dataset, only use first anchor
    if trade_forward_days >= total_days * 0.8:  # 80% threshold
        print(f"trade_forward_days ({trade_forward_days}) is large relative to dataset ({total_days} days)")
        print(f"Using single anchor approach: screen once, trade for {trade_forward_days} days")
        return [first_valid_anchor]
    
    # Create anchors: start with first Monday, then roll forward by trade_forward_days
    current_anchor = first_valid_anchor
    while current_anchor <= dataset_end:
        anchors.append(current_anchor)
        
        # Calculate next anchor: current + trade_forward_days
        next_anchor = current_anchor + pd.Timedelta(days=trade_forward_days)
        
        # Find the closest trading day to next_anchor (within a few days)
        if next_anchor > dataset_end:
            break
            
        # Find the closest trading day to next_anchor (within a few days)
        closest_trading_day = None
        for direction in [0, 1, -1, 2, -2, 3, -3]:
            check_date = (next_anchor + pd.Timedelta(days=direction)).date()
            # Check if any timestamp on this DATE exists in index
            day_mask = index.date == check_date
            if day_mask.any():
                closest_trading_day = index[day_mask][0]  # First timestamp of that day
                break

        if closest_trading_day is None or closest_trading_day > dataset_end:
            break
            
        current_anchor = closest_trading_day
    
    print(f"Created {len(anchors)} anchors with {trade_forward_days} day trading periods")
    print(f"First anchor: {first_valid_anchor} (after {train_lookback_days} days of training data)")
    return anchors


def run_backtest(df: pd.DataFrame, pairs: List[Tuple[str, str]], config: BacktestConfig):
    price_map = wide_prices(df, price=config.data.price_field, rth_only=config.data.rth_only)
    if not price_map:
        raise ValueError("No prices available after preprocessing")

    all_index = None
    for s in price_map.values():
        all_index = s.index if all_index is None else all_index.union(s.index)
    index = pd.DatetimeIndex(sorted(all_index))

    bar_minutes = config.screening.bar_minutes
    train_lookback_days = config.engine.train_lookback_days
    trade_forward_days = config.engine.trade_forward_days

    anchors = _weekly_anchors(index, train_lookback_days, trade_forward_days)
    ledger_rows: List[Dict] = []
    
    cash = pd.Series(float(config.execution.starting_cash), index=index, dtype='float64')
    equity = pd.Series(0.0, index=index, dtype='float64')
    
    z_calculator = AdaptiveZScore(
        tau_minutes=config.signals.tau_minutes,
        warmup_min_bars=config.signals.warmup_min_bars,
        sigma_floor=config.signals.sigma_floor,
        reset_each_session=config.signals.reset_each_session,
        method=config.signals.z_score_method,
        use_adaptive_window=config.signals.use_adaptive_window,
        tau_scale_factor=config.signals.tau_scale_factor,
        min_tau_minutes=config.signals.min_tau_minutes,
        max_tau_minutes=config.signals.max_tau_minutes
    )

    open_positions: Dict[str, Dict] = {}
    last_screen: Dict[str, Dict] = {}
    previous_params: Dict[str, Dict] = {}

    for anchor in anchors:
        train_end = anchor - pd.Timedelta(minutes=1)
        train_start = train_end - pd.Timedelta(days=train_lookback_days)
        trade_end = anchor + pd.Timedelta(days=trade_forward_days)

        train_prices = {k: v[(v.index >= train_start) & (v.index <= train_end)] for k, v in price_map.items()}
        screen_df = screen_pairs(pairs, train_prices, config)
        last_screen = {r["pair"]: r for _, r in screen_df.iterrows()}

        tradables = [r for _, r in screen_df.iterrows() if r["cointegrated"]]

        frozen = {}
        for r in tradables:
            pair = r["pair"]
            a, b = r["symbol_x"], r["symbol_y"]
            
            if pair in previous_params:
                old_alpha = previous_params[pair].get("alpha")
                old_beta = previous_params[pair].get("beta")
                if (old_alpha != r["alpha"]) or (old_beta != r["beta"]):
                    print(f"Parameters changed for {pair}: alpha {old_alpha:.2f}->{r['alpha']:.2f}, beta {old_beta:.2f}->{r['beta']:.2f} - resetting z-score")
                    z_calculator.reset_pair_stats(pair)
            
            tau_half_bars = int(np.ceil((r["half_life_minutes"] or np.inf) / bar_minutes)) if np.isfinite(r["half_life_minutes"]) else np.inf
            
            # *** FIX: Set adaptive tau BEFORE seeding ***
            if config.signals.use_adaptive_window and np.isfinite(r["half_life_minutes"]):
                z_calculator.set_pair_tau_minutes(pair, r["half_life_minutes"])
                adaptive_tau = z_calculator.get_pair_tau_minutes(pair)
                print(f"Set adaptive tau for {pair}: half_life={r['half_life_minutes']:.1f}m -> tau={adaptive_tau:.1f}m")
            
            frozen[pair] = {
                "alpha": r["alpha"],
                "beta": r["beta"],
                "sigma": r.get("sigma", 1.0),
                "tau_half_bars": tau_half_bars,
                "half_life_minutes": r["half_life_minutes"],
                "x": price_map[a],
                "y": price_map[b],
                "a": a,
                "b": b,
            }
            
            previous_params[pair] = {
                "alpha": r["alpha"],
                "beta": r["beta"]
            }

            # *** Now seeding will use the correct adaptive tau ***
            x_train = train_prices.get(a)
            y_train = train_prices.get(b)
            if x_train is not None and y_train is not None and len(x_train) > 0 and len(y_train) > 0:
                idx_train = x_train.index.intersection(y_train.index)
                if len(idx_train) > 0:
                    x_aligned = x_train.reindex(idx_train).ffill().astype(float)
                    y_aligned = y_train.reindex(idx_train).ffill().astype(float)
                    u_train = y_aligned - (r["alpha"] + r["beta"] * x_aligned)
                    try:
                        z_calculator.seed_from_training(pair, u_train)
                    except Exception:
                        pass

        trade_index = index[(index >= anchor) & (index < trade_end)]
        in_trade_bars: Dict[str, int] = {k: 0 for k in frozen.keys()}

        for t in trade_index:
            # Descreen check
            for pair, pos in list(open_positions.items()):
                if pair not in last_screen or not last_screen[pair].get("cointegrated", False):
                    a = pos["a"]; b = pos["b"]
                    price_y = price_map[b].get(t, np.nan)
                    price_x = price_map[a].get(t, np.nan)
                    
                    if np.isnan(price_y) or np.isnan(price_x):
                        continue
                    
                    fill = simulate_fill(price_y, price_x, pos["qty_y"], pos["qty_x"],
                                       config.execution.fee_bps, config.execution.slip_bps)
                    
                    pnl = (price_y - pos["entry_price_y"]) * pos["qty_y"] + \
                          (price_x - pos["entry_price_x"]) * pos["qty_x"] - (fill.fee + fill.slip)
                    
                    u_exit = price_y - (pos["alpha"] + pos["beta"] * price_x)
                    spread_change = u_exit - pos["u_entry"]
                    
                    ledger_rows.append({
                        "pair": pair,
                        "open_time": pos["open_time"],
                        "close_time": t,
                        "side": pos["side"],
                        "alpha": pos["alpha"],
                        "beta": pos["beta"],
                        "half_life_minutes": pos.get("half_life_minutes"),
                        "qty_y": pos["qty_y"],
                        "qty_x": pos["qty_x"],
                        "entry_price_y": pos["entry_price_y"],
                        "entry_price_x": pos["entry_price_x"],
                        "exit_price_y": price_y,
                        "exit_price_x": price_x,
                        "u_entry": pos["u_entry"],
                        "u_exit": u_exit,
                        "spread_change": spread_change,
                        "entry_z": pos["entry_z"],
                        "exit_z": np.nan,
                        "z_change": np.nan,
                        "pnl": pnl,
                        "fees": fill.fee + fill.slip,
                        "reason": "descreen",
                    })
                    
                    cash.loc[t:] += pnl + pos['reserved_cash']
                    equity.loc[t:] += pnl
                    del open_positions[pair]

            for pair, fr in frozen.items():
                a, b = fr["a"], fr["b"]
                price_y = fr["y"].get(t, np.nan)
                price_x = fr["x"].get(t, np.nan)
                if np.isnan(price_y) or np.isnan(price_x):
                    continue
                u_t = price_y - (fr["alpha"] + fr["beta"] * price_x)
                z_t = z_calculator.update_and_get_z(pair, u_t, t)

                if pair in open_positions:
                    rsn = exit_reason(z_t, config.signals.z_out, config.signals.z_max)
                    in_trade_bars[pair] += 1
                    if fr["tau_half_bars"] and np.isfinite(fr["tau_half_bars"]) and in_trade_bars[pair] >= 2 * fr["tau_half_bars"]:
                        rsn = rsn or "time_stop"
                    
                    if rsn:
                        pos = open_positions[pair]
                        fill = simulate_fill(price_y, price_x, pos["qty_y"], pos["qty_x"],
                                           config.execution.fee_bps, config.execution.slip_bps)
                        
                        pnl = (price_y - pos["entry_price_y"]) * pos["qty_y"] + \
                              (price_x - pos["entry_price_x"]) * pos["qty_x"] - (fill.fee + fill.slip)
                        
                        u_exit = price_y - (pos["alpha"] + pos["beta"] * price_x)
                        spread_change = u_exit - pos["u_entry"]
                        z_change = z_t - pos["entry_z"] if not np.isnan(z_t) and not np.isnan(pos["entry_z"]) else np.nan
                        
                        ledger_rows.append({
                            "pair": pair,
                            "open_time": pos["open_time"],
                            "close_time": t,
                            "side": pos["side"],
                            "alpha": pos["alpha"],
                            "beta": pos["beta"],
                            "half_life_minutes": pos.get("half_life_minutes"),
                            "qty_y": pos["qty_y"],
                            "qty_x": pos["qty_x"],
                            "entry_price_y": pos["entry_price_y"],
                            "entry_price_x": pos["entry_price_x"],
                            "exit_price_y": price_y,
                            "exit_price_x": price_x,
                            "u_entry": pos["u_entry"],
                            "u_exit": u_exit,
                            "spread_change": spread_change,
                            "entry_z": pos["entry_z"],
                            "exit_z": z_t,
                            "z_change": z_change,
                            "pnl": pnl,
                            "fees": fill.fee + fill.slip,
                            "reason": rsn,
                        })
                        
                        cash.loc[t:] += pnl + pos['reserved_cash']
                        equity.loc[t:] += pnl
                        del open_positions[pair]
                    continue

                # ENTRY SIGNAL
                side = entry_signal(
                    z_t,
                    config.signals.z_in,
                    config.signals.z_max,
                    getattr(config.signals, "z_cap", None),
                )
                if side == 0:
                    continue
                
                # KELLY-BASED POSITION SIZING
                available_cash = cash.loc[t]
                kelly_frac = 0.0
                fixed_adjustment = 3.0

                if config.kelly.enabled:
                    lookback = config.kelly.lookback_bars
                    y_hist = fr["y"].iloc[max(0, fr["y"].index.get_loc(t) - lookback):fr["y"].index.get_loc(t)+1]
                    x_hist = fr["x"].iloc[max(0, fr["x"].index.get_loc(t) - lookback):fr["x"].index.get_loc(t)+1]
                    spread = y_hist - (fr["alpha"] + fr["beta"] * x_hist)
                    if len(spread) > 5:
                        kelly_frac = kelly_from_spread(spread) * config.kelly.scale
                        kelly_frac = clamp(
                            kelly_frac,
                            config.kelly.min_frac,
                            min(config.kelly.max_frac, config.execution.max_allocation_pct)
                        )

                if kelly_frac == 0.0:
                    # fallback to fixed position sizing
                    target_notional = config.execution.notional_per_trade
                    max_notional_from_allocation = available_cash * config.execution.max_allocation_pct
                    effective_notional = min(target_notional, max_notional_from_allocation)
                else:
                    effective_notional = available_cash * kelly_frac * fixed_adjustment

                
                # KEY FIX: Calculate gross exposure accounting for beta
                # For a pairs trade, the gross notional exposure is:
                # |Y notional| + |X notional| = |Y notional| * (1 + |beta|)
                # 
                # We want TOTAL gross exposure = effective_notional
                # So: Y_notional * (1 + |beta|) = effective_notional
                # Therefore: Y_notional = effective_notional / (1 + |beta|)
                
                gross_leverage = 1.0 + abs(fr["beta"])
                y_notional = effective_notional / gross_leverage
                
                # Now calculate quantities based on this beta-adjusted notional
                if side == 1:  # long spread (long Y, short X)
                    qty_y = y_notional / price_y
                    qty_x = -fr["beta"] * qty_y
                else:  # short spread (short Y, long X)
                    qty_y = -y_notional / price_y
                    qty_x = -fr["beta"] * qty_y

                # Apply volatility scaling if configured
                if config.execution.vol_scale and config.execution.spread_sigma_target:
                    qty_y, qty_x = apply_vol_scale(qty_y, qty_x, fr["sigma"], 
                                                   config.execution.spread_sigma_target)
                
                fill = simulate_fill(price_y, price_x, qty_y, qty_x,
                                   config.execution.fee_bps, config.execution.slip_bps)
                
                # Calculate actual capital required (long leg + fees)
                if side == 1:  # long spread (long Y, short X)
                    capital_required = abs(qty_y * price_y) + fill.fee + fill.slip
                else:  # short spread (short Y, long X)
                    capital_required = abs(qty_x * price_x) + fill.fee + fill.slip
                    
                # Check if we have enough cash
                if capital_required > available_cash:
                    continue

                # SUBTRACT PRICE OF LONG LEG FROM CASH
                cash.loc[t:] -= capital_required
                
                u_entry = price_y - (fr["alpha"] + fr["beta"] * price_x)
                
                open_positions[pair] = {
                    "open_time": t,
                    "side": side,
                    "alpha": fr["alpha"],
                    "beta": fr["beta"],
                    "qty_y": qty_y,
                    "qty_x": qty_x,
                    "entry_z": z_t,
                    "entry_price_y": price_y,
                    "entry_price_x": price_x,
                    "u_entry": u_entry,
                    "half_life_minutes": fr.get("half_life_minutes"),
                    "a": a,
                    "b": b,
                    "reserved_cash": capital_required, # STORE PRICE OF LONG ENTRY LEG
                }
                in_trade_bars[pair] = 0

    # Force close remaining positions
    if open_positions:
        print(f"Force closing {len(open_positions)} open positions at end of backtest")
        final_time = index[-1] if len(index) > 0 else None
        if final_time is not None:
            for pair, pos in open_positions.items():
                a, b = pos["a"], pos["b"]
                price_y = price_map[b].get(final_time, np.nan)
                price_x = price_map[a].get(final_time, np.nan)
                
                if not (np.isnan(price_y) or np.isnan(price_x)):
                    fill = simulate_fill(price_y, price_x, pos["qty_y"], pos["qty_x"],
                                       config.execution.fee_bps, config.execution.slip_bps)
                    
                    pnl = (price_y - pos["entry_price_y"]) * pos["qty_y"] + \
                          (price_x - pos["entry_price_x"]) * pos["qty_x"] - (fill.fee + fill.slip)
                    
                    u_exit = price_y - (pos["alpha"] + pos["beta"] * price_x)
                    spread_change = u_exit - pos["u_entry"]
                    
                    ledger_rows.append({
                        "pair": pair,
                        "open_time": pos["open_time"],
                        "close_time": final_time,
                        "side": pos["side"],
                        "alpha": pos["alpha"],
                        "beta": pos["beta"],
                        "half_life_minutes": pos.get("half_life_minutes"),
                        "qty_y": pos["qty_y"],
                        "qty_x": pos["qty_x"],
                        "entry_price_y": pos["entry_price_y"],
                        "entry_price_x": pos["entry_price_x"],
                        "exit_price_y": price_y,
                        "exit_price_x": price_x,
                        "u_entry": pos["u_entry"],
                        "u_exit": u_exit,
                        "spread_change": spread_change,
                        "entry_z": pos["entry_z"],
                        "exit_z": np.nan,
                        "z_change": np.nan,
                        "pnl": pnl,
                        "fees": fill.fee + fill.slip,
                        "reason": "end_of_backtest",
                    })
                    
                    cash.loc[final_time:] += pnl + pos['reserved_cash']
                    equity.loc[final_time:] += pnl

    ledger = pd.DataFrame(ledger_rows)
    results = {
        "ledger": ledger,
        "equity": equity,
        "cash": cash,
    }
    return results