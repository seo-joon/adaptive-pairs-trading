from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import yaml


@dataclass
class DataConfig:
    rth_only: bool = True
    calendar: str = "US_EQ"
    price_field: str = "close"


@dataclass
class ScreeningConfig:
    min_obs: int = 80
    bar_minutes: int = 1
    hl_min_minutes: int = 6 # (REVERSION HALF LIFE PARAMETERS)
    hl_max_minutes: int = 2000 # 180 (trading longer time frames)
    use_kpss: bool = False


@dataclass
class SignalsConfig:
    z_in: float = 1.4
    z_out: float = 0.6
    z_max: float = 3.0
    # Optional entry ceiling to avoid entering too close to z_max
    z_cap: Optional[float] = 3.0
    # Z-score calculation parameters
    tau_minutes: float = 600 # ROLLING WINDOW FOR SPREAD (120) 
    warmup_min_bars: int = 30   # minimum bars before trading # (not super important becuase we prefill the rolling window)
    sigma_floor: float = 1e-8   # minimum sigma to avoid division by zero
    reset_each_session: bool = False  # keep previous day's data for immediate trading
    z_score_method: str = "rolling"  # "rolling", "ewma", or "hybrid"

    # Adaptive window sizing parameters
    use_adaptive_window: bool = True  # Enable adaptive window sizing based on half-lives
    tau_scale_factor: float = 3.0  # Multiply half-life by this factor to get tau_minutes
    min_tau_minutes: float = 60  # Minimum tau_minutes (1 hour)
    max_tau_minutes: float = 7200  # Maximum tau_minutes (5 days)


@dataclass
class ExecutionConfig:
    fill: str = "close"  # "close" or "next_open"
    fee_bps: float = 0.2 # 0.2
    slip_bps: float = 0.3 # 0.3
    notional_per_trade: float = 50_000 # 10_000 # (bounds trades, min(allocation by pct, notional))
    vol_scale: bool = False
    spread_sigma_target: Optional[float] = None
    # Cash management
    starting_cash: float = 100_000  # Starting capital
    max_allocation_pct: float = 0.5  # 0.1 Max 10% of cash per trade
    

@dataclass
class KellyConfig:
    enabled: bool = True
    scale: float = 1.25           # fractional Kelly (0.25–0.5 commonly used here)
    lookback_bars: int = 90       # number of bars for μ, σ estimation
    min_frac: float = 0.0
    max_frac: float = 0.35        # hard cap per position



@dataclass
class EngineConfig:
    train_lookback_days: int = 60 # 21
    trade_forward_days: int = 5 # 5
    seed: int = 42
    max_concurrent_positions: Optional[int] = None
    allow_reentry_same_day: bool = True


@dataclass
class BacktestConfig:
    data: DataConfig = field(default_factory=DataConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    extras: Dict[str, Any] = field(default_factory=dict)
    
    kelly: KellyConfig = field(default_factory=KellyConfig)


def _to_dataclass(cls, d: Dict[str, Any]):
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_config(config: Optional[str | Dict[str, Any]] = None) -> BacktestConfig:
    if config is None:
        return BacktestConfig()
    if isinstance(config, dict):
        raw = config
    else:
        with open(config, "r") as f:
            raw = yaml.safe_load(f) or {}

    data = _to_dataclass(DataConfig, raw.get("data", {}))
    screening = _to_dataclass(ScreeningConfig, raw.get("screening", {}))
    signals = _to_dataclass(SignalsConfig, raw.get("signals", {}))
    execution = _to_dataclass(ExecutionConfig, raw.get("execution", {}))
    engine = _to_dataclass(EngineConfig, raw.get("engine", {}))

    extras = {k: v for k, v in raw.items() if k not in {"data", "screening", "signals", "execution", "engine"}}
    
    kelly = _to_dataclass(KellyConfig, raw.get("kelly", {}))

    return BacktestConfig(
        data=data,
        screening=screening,
        signals=signals,
        execution=execution,
        engine=engine,
        extras=extras,
        kelly=kelly,
    )


