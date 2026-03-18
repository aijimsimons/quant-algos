"""Bollinger Bands strategy."""

from typing import Callable

import polars as pl


def bollinger_strategy(
    period: int = 20,
    num_std: float = 2.0
) -> Callable:
    """Bollinger Bands mean reversion."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        close = pl.col("close")
        mean = close.rolling_mean(window_size=period)
        std = close.rolling_std(window_size=period)
        
        upper = mean + (std * num_std)
        lower = mean - (std * num_std)
        
        df = data.with_columns([
            upper.alias("bb_upper"),
            lower.alias("bb_lower"),
        ])
        
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        close = latest["close"]
        upper = latest["bb_upper"]
        lower = latest["bb_lower"]
        
        if close > upper:
            return [{
                "symbol": latest["symbol"],
                "size": -1.0,
                "price": close,
            }]
        elif close < lower:
            return [{
                "symbol": latest["symbol"],
                "size": 1.0,
                "price": close,
            }]
        
        return []
    
    return generate_signals
