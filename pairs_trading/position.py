from __future__ import annotations

from typing import Dict, Tuple


def hedge_quantities(price_y: float, price_x: float, beta: float, notional: float, side: int) -> Tuple[float, float]:
    # side: 1 -> long y/short x; -1 -> short y/long x
    q_y = (notional / price_y) * side
    q_x = (beta * notional / price_x) * (-side)
    return q_y, q_x


def apply_vol_scale(q_y: float, q_x: float, spread_sigma: float, target_sigma: float | None) -> Tuple[float, float]:
    if target_sigma and spread_sigma and spread_sigma > 0:
        scale = target_sigma / spread_sigma
        return q_y * scale, q_x * scale
    return q_y, q_x


def kelly_fraction(expected_return: float, volatility: float) -> float:
    if volatility <= 0:
        return 0
    return expected_return / (volatility ** 2)

