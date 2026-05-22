from __future__ import annotations

from dataclasses import dataclass

from .utils import to_bps


@dataclass
class Fill:
    price_y: float
    price_x: float
    fee: float
    slip: float


def simulate_fill(price_y: float, price_x: float, qty_y: float, qty_x: float,
                  fee_bps: float, slip_bps: float) -> Fill:
    """Calculate fees and slippage based on actual position sizes."""
    notional_y = abs(qty_y * price_y)
    notional_x = abs(qty_x * price_x)
    total_notional = notional_y + notional_x
    
    fee = total_notional * to_bps(fee_bps)
    slip = total_notional * to_bps(slip_bps)
    
    return Fill(price_y=price_y, price_x=price_x, fee=fee, slip=slip)


