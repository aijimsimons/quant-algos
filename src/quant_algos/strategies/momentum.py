"""Momentum-based trading strategies."""

from typing import Callable

import polars as pl


def momentum_strategy(period: int = 20, threshold: float = 0.002) -> Callable:
    """Simple momentum strategy."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        # Calculate returns
        df = data.with_columns(
            (pl.col("close") / pl.col("close").shift(period) - 1).alias(f"return_{period}d")
        )
        
        # Generate signals
        df = df.with_columns(
            pl.when(pl.col(f"return_{period}d") > threshold).then(1.0)
             .when(pl.col(f"return_{period}d") < -threshold).then(-1.0)
             .otherwise(0.0)
             .alias("signal")
        )
        
        # Get latest row as dict
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        signal = latest["signal"]
        
        if signal != 0:
            # Scale position size to affordable amount (2% of capital)
            price = latest["close"]
            position_value = capital * 0.02  # 2% of capital
            size = position_value / price
            
            return [{
                "symbol": latest["symbol"],
                "size": signal * size,
                "price": price,
            }]
        
        return []
    
    return generate_signals
