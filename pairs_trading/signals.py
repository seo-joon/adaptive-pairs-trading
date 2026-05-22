from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class AdaptiveZScore:
    """Robust z-score calculator for pairs trading with multiple approaches."""
    
    def __init__(self, tau_minutes: float, warmup_min_bars: int, sigma_floor: float, reset_each_session: bool = False, method: str = "rolling", 
                 use_adaptive_window: bool = False, tau_scale_factor: float = 2.5, min_tau_minutes: float = 60, max_tau_minutes: float = 7200):
        self.base_tau_minutes = tau_minutes  # Default/fallback tau_minutes
        self.warmup_min_bars = warmup_min_bars
        self.sigma_floor = sigma_floor
        self.reset_each_session = reset_each_session
        self.method = method  # "ewma", "rolling", or "hybrid"
        self.use_adaptive_window = use_adaptive_window
        self.tau_scale_factor = tau_scale_factor
        self.min_tau_minutes = min_tau_minutes
        self.max_tau_minutes = max_tau_minutes
        self.lambda_ = 1 - np.exp(-1 / tau_minutes)  # Base lambda for fallback
        self.stats: Dict[str, Dict] = {}  # per-pair stats
        self.spread_history: Dict[str, List] = {}  # Store recent spreads for rolling stats
        self.pair_tau_minutes: Dict[str, float] = {}  # Store pair-specific tau_minutes
    
    def _is_session_open(self, t: pd.Timestamp) -> bool:
        """Check if this is the first bar of a new trading session."""
        if not self.reset_each_session:
            return False
        # Simple check: if we're at 9:30 AM ET (13:30 UTC) on a weekday
        ny_time = t.tz_convert('America/New_York') if t.tz else t.tz_localize('UTC').tz_convert('America/New_York')
        return (ny_time.weekday() < 5 and 
                ny_time.hour == 9 and 
                ny_time.minute == 30)
    
    def set_pair_tau_minutes(self, pair: str, half_life_minutes: float):
        """Set pair-specific tau_minutes based on half-life."""
        if not self.use_adaptive_window or not np.isfinite(half_life_minutes):
            print('Using base tau minutes')
            self.pair_tau_minutes[pair] = self.base_tau_minutes
            return
        
        # Calculate adaptive tau_minutes: tau = scale_factor * half_life
        adaptive_tau = self.tau_scale_factor * half_life_minutes
        
        # Apply min/max constraints
        adaptive_tau = max(adaptive_tau, self.min_tau_minutes)
        adaptive_tau = min(adaptive_tau, self.max_tau_minutes)
        
        self.pair_tau_minutes[pair] = adaptive_tau
        
        # Update lambda for this pair if using EWMA or hybrid
        if self.method in ["ewma", "hybrid"]:
            if pair not in self.stats:
                self.stats[pair] = {'m1': None, 'm2': None, 'n': 0, 'last_session': None}
            # Store lambda in stats for this pair
            self.stats[pair]['lambda'] = 1 - np.exp(-1 / adaptive_tau)
    
    def get_pair_tau_minutes(self, pair: str) -> float:
        """Get tau_minutes for a specific pair."""
        return self.pair_tau_minutes.get(pair, self.base_tau_minutes)
    
    def get_pair_lambda(self, pair: str) -> float:
        """Get lambda (decay rate) for a specific pair."""
        if self.method not in ["ewma", "hybrid"]:
            return self.lambda_
        
        pair_tau = self.get_pair_tau_minutes(pair)
        return 1 - np.exp(-1 / pair_tau)
    
    def update_and_get_z(self, pair: str, u_t: float, t: pd.Timestamp) -> float:
        """Update stats and return z-score for this bar using the specified method."""
        if pair not in self.stats:
            self.stats[pair] = {'m1': None, 'm2': None, 'n': 0, 'last_session': None}
            self.spread_history[pair] = []
        
        stats = self.stats[pair]
        
        # Check for session reset
        if self._is_session_open(t):
            stats['m1'] = None
            stats['m2'] = None
            stats['n'] = 0
            stats['last_session'] = t.date()
            self.spread_history[pair] = []
        
        # Add current spread to history
        self.spread_history[pair].append(u_t)
        
        # *** FIX: Use pair-specific tau for memory management ***
        pair_tau = self.get_pair_tau_minutes(pair)
        
        # Keep only recent history (last 2 * pair_tau_minutes bars)
        # Cap at 10000 to prevent unbounded memory even if base_tau is huge
        #  # (be careful here, might need to adjust later if we are trading longer time horizons where half life is much longer)
        max_history = int(2 * min(pair_tau, self.max_tau_minutes))
        
        if len(self.spread_history[pair]) > max_history:
            self.spread_history[pair] = self.spread_history[pair][-max_history:]
        
        # Calculate z-score based on method
        if self.method == "rolling":
            return self._rolling_z_score(pair, u_t)
        elif self.method == "ewma":
            return self._ewma_z_score(pair, u_t)
        elif self.method == "hybrid":
            return self._hybrid_z_score(pair, u_t)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def seed_from_training(self, pair: str, spreads: pd.Series):
        """Seed internal statistics from recent training spreads.

        This removes cold-start behavior at trade start by initializing the
        rolling window and/or EWMA accumulators using the last W spreads from
        the training window, where W = min(tau_minutes, available_bars).

        Parameters
        ----------
        pair : str
            The pair identifier key.
        spreads : pd.Series
            Training period spreads (y - (alpha + beta * x)), indexed by time.
        """
        if spreads is None or len(spreads) == 0:
            return

        # Ensure per-pair containers exist
        if pair not in self.stats:
            self.stats[pair] = {'m1': None, 'm2': None, 'n': 0, 'last_session': None}
        if pair not in self.spread_history:
            self.spread_history[pair] = []

        clean = pd.Series(spreads).dropna()
        if len(clean) == 0:
            return

        # Use pair-specific tau_minutes if available, otherwise fall back to base
        pair_tau = self.get_pair_tau_minutes(pair)
        window = int(pair_tau) if np.isfinite(pair_tau) else len(clean)
        w = max(1, min(window, len(clean)))
        last_w = clean.iloc[-w:]

        # Common seed statistics
        mu0 = float(np.mean(last_w))
        # Use ddof=1 when possible (w>1) to avoid biased zero variance
        if w > 1:
            var0 = float(np.var(last_w, ddof=1))
        else:
            var0 = 0.0
        sigma0 = max(np.sqrt(max(var0, 0.0)), self.sigma_floor)

        stats = self.stats[pair]

        if self.method == "rolling":
            # Prefill rolling history so rolling z can be computed immediately
            self.spread_history[pair] = list(last_w.values)
            # Set tracking stats to reflect seeded history
            stats['m1'] = mu0
            stats['m2'] = var0 + mu0 * mu0
            stats['n'] = len(self.spread_history[pair])
        elif self.method == "ewma":
            # EWMA uses recursive moments; seed level and second moment so that
            # initial variance equals var0
            stats['m1'] = mu0
            stats['m2'] = var0 + mu0 * mu0
            stats['n'] = w
            # History not required for EWMA, keep it minimal to avoid memory bloat
            self.spread_history[pair] = []
        elif self.method == "hybrid":
            # Hybrid: rolling mean + EWMA volatility of residuals
            self.spread_history[pair] = list(last_w.values)
            stats['m1'] = mu0
            # Seed EWMA variance accumulator with sample variance of residuals
            stats['m2'] = max(var0, self.sigma_floor * self.sigma_floor)
            stats['n'] = len(self.spread_history[pair])
        else:
            # Unknown method; do nothing
            return
    
    def _rolling_z_score(self, pair: str, u_t: float) -> float:
        """Calculate z-score using rolling window statistics."""
        history = self.spread_history[pair]
        stats = self.stats[pair]
        
        if len(history) < self.warmup_min_bars:
            return float('nan')
        
        # Use pair-specific tau_minutes for window size
        pair_tau = self.get_pair_tau_minutes(pair)
        window_size = min(len(history), int(pair_tau))
        recent_spreads = history[-window_size:]
        
        mean_t = np.mean(recent_spreads)
        std_t = np.std(recent_spreads, ddof=1)
        std_t = max(std_t, self.sigma_floor)
        
        # Update stats for tracking
        stats['m1'] = mean_t
        stats['m2'] = std_t * std_t + mean_t * mean_t
        stats['n'] = len(history)
        
        return (u_t - mean_t) / std_t
    
    def _ewma_z_score(self, pair: str, u_t: float) -> float:
        """Calculate z-score using EWMA statistics (original method)."""
        stats = self.stats[pair]
        
        # Get pair-specific lambda
        pair_lambda = self.get_pair_lambda(pair)
        
        # Initialize or update EWMA stats
        if stats['m1'] is None:
            stats['m1'] = u_t
            stats['m2'] = u_t * u_t
            stats['n'] = 1
        else:
            stats['m1'] = (1 - pair_lambda) * stats['m1'] + pair_lambda * u_t
            stats['m2'] = (1 - pair_lambda) * stats['m2'] + pair_lambda * (u_t * u_t)
            stats['n'] += 1
        
        # Compute variance and sigma
        var_t = max(stats['m2'] - stats['m1'] * stats['m1'], 0.0)
        sigma_t = max(np.sqrt(var_t), self.sigma_floor)
        
        if stats['n'] < self.warmup_min_bars or sigma_t <= 0:
            return float('nan')
        else:
            return (u_t - stats['m1']) / sigma_t
    
    def _hybrid_z_score(self, pair: str, u_t: float) -> float:
        """Calculate z-score using hybrid approach: rolling mean with EWMA volatility."""
        history = self.spread_history[pair]
        stats = self.stats[pair]
        
        if len(history) < self.warmup_min_bars:
            return float('nan')
        
        # Use pair-specific tau_minutes for rolling mean window
        pair_tau = self.get_pair_tau_minutes(pair)
        window_size = min(len(history), int(pair_tau))
        recent_spreads = history[-window_size:]
        mean_t = np.mean(recent_spreads)
        
        # Use pair-specific lambda for EWMA volatility
        pair_lambda = self.get_pair_lambda(pair)
        if stats['m2'] is None:
            stats['m2'] = (u_t - mean_t) ** 2
        else:
            stats['m2'] = (1 - pair_lambda) * stats['m2'] + pair_lambda * ((u_t - mean_t) ** 2)
        
        std_t = max(np.sqrt(stats['m2']), self.sigma_floor)
        
        # Update stats for tracking
        stats['m1'] = mean_t
        stats['n'] = len(history)
        
        return (u_t - mean_t) / std_t
    
    def reset_pair_stats(self, pair: str):
        """Reset statistics for a specific pair (used when alpha/beta parameters change)."""
        if pair in self.stats:
            self.stats[pair] = {'m1': None, 'm2': None, 'n': 0, 'last_session': None}
        if pair in self.spread_history:
            self.spread_history[pair] = []
        print(f"Reset z-score statistics for pair: {pair}")


def compute_spread_and_z(
    x: pd.Series,
    y: pd.Series,
    alpha: float,
    beta: float,
    mu_u: float,
    sigma_u: float,
) -> Tuple[pd.Series, pd.Series]:
    """Legacy function - kept for compatibility but not used in new implementation."""
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).ffill().astype(float)
    y = y.reindex(idx).ffill().astype(float)
    u = y - (alpha + beta * x)
    if sigma_u <= 0 or np.isnan(sigma_u):
        z = pd.Series(np.nan, index=u.index)
    else:
        z = (u - mu_u) / sigma_u
    return u, z


def compute_adaptive_z_score(
    x: pd.Series,
    y: pd.Series,
    alpha: float,
    beta: float,
    z_calculator: AdaptiveZScore,
    pair: str
) -> Tuple[pd.Series, pd.Series]:
    """Compute spread and adaptive z-scores using EWMA stats."""
    idx = x.index.intersection(y.index)
    x = x.reindex(idx).ffill().astype(float)
    y = y.reindex(idx).ffill().astype(float)
    u = y - (alpha + beta * x)
    
    # Compute adaptive z-scores
    z_values = []
    for i, (t, u_val) in enumerate(zip(u.index, u.values)):
        if np.isnan(u_val):
            z_values.append(float('nan'))
        else:
            z_val = z_calculator.update_and_get_z(pair, u_val, t)
            z_values.append(z_val)
    
    z = pd.Series(z_values, index=u.index)
    return u, z


def entry_signal(z_t: float, z_in: float, z_max: float, z_cap: float | None = None) -> int:
    if np.isnan(z_t):
        return 0
    abs_z = abs(z_t)
    # Use z_cap if provided, otherwise default to z_max
    ceiling = z_cap if (z_cap is not None) else z_max
    if z_in < abs_z < ceiling:
        return -1 if z_t > 0 else 1  # -1: short y/long x, 1: long y/short x
    return 0


def exit_reason(z_t: float, z_out: float, z_max: float) -> str | None:
    if np.isnan(z_t):
        return None
    if abs(z_t) <= z_out:
        return "mean_revert"
    if abs(z_t) > z_max:
        return "adverse"
    return None


