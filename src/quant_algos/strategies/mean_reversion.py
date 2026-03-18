"""Mean reversion strategy using Z-score."""

from typing import Callable

import polars as pl


def mean_reversion_strategy(
    period: int = 20,
    z_threshold: float = 2.0
) -> Callable:
    """Mean reversion strategy using Z-score."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        # Calculate Z-score
        close = pl.col("close")
        mean = close.rolling_mean(window_size=period)
        std = close.rolling_std(window_size=period)
        z_score = (close - mean) / std
        
        df = data.with_columns(z_score.alias("z_score"))
        
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        z = latest["z_score"]
        
        if z > z_threshold:
            return [{
                "symbol": latest["symbol"],
                "size": -1.0,
                "price": latest["close"],
            }]
        elif z < -z_threshold:
            return [{
                "symbol": latest["symbol"],
                "size": 1.0,
                "price": latest["close"],
            }]
        
        return []
    
    return generate_signals
