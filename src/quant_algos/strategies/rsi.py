"""RSI-based strategy."""

from typing import Callable

import polars as pl


def rsi_strategy(
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> Callable:
    """RSI-based strategy."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        delta = pl.col("close").diff()
        gain = delta.clip(min=0)
        loss = -delta.clip(max=0)
        
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df = data.with_columns(rsi.alias("rsi"))
        
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        rsi_val = latest["rsi"]
        
        if rsi_val > overbought:
            return [{
                "symbol": latest["symbol"],
                "size": -1.0,
                "price": latest["close"],
            }]
        elif rsi_val < oversold:
            return [{
                "symbol": latest["symbol"],
                "size": 1.0,
                "price": latest["close"],
            }]
        
        return []
    
    return generate_signals
