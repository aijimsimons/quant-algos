"""Strategy registry and common strategies."""

from typing import Callable

import polars as pl


# Import all strategies
from .momentum import momentum_strategy
from .mean_reversion import mean_reversion_strategy
from .bollinger import bollinger_strategy
from .rsi import rsi_strategy
from .regime import regime_detection_strategy
from .pairs import pairs_strategy
from .volatility import volatility_arbitrage

# Strategy registry
_strategies: dict[str, Callable] = {
    "momentum": momentum_strategy,
    "mean_reversion": mean_reversion_strategy,
    "bollinger": bollinger_strategy,
    "rsi": rsi_strategy,
    "regime_detection": regime_detection_strategy,
    "pairs": pairs_strategy,
    "volatility": volatility_arbitrage,
}


def get_strategy(name: str) -> Callable:
    """Get a registered strategy."""
    if name not in _strategies:
        raise ValueError(f"Strategy '{name}' not found. Available: {list(_strategies.keys())}")
    return _strategies[name]


def list_strategies() -> list[str]:
    """List all registered strategies."""
    return list(_strategies.keys())
