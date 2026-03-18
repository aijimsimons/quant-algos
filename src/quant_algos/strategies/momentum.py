"""Momentum-based trading strategies."""

from typing import Callable

import polars as pl


def momentum_strategy() -> Callable:
    """Simple momentum strategy."""
    def generate_signals(df: pl.DataFrame, positions: list, capital: float) -> list:
        # Calculate returns
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("return_20d")
        )
        
        # Generate signals based on momentum
        df = df.with_columns(
            pl.when(pl.col("return_20d") > 0.05).then(1.0)
             .when(pl.col("return_20d") < -0.05).then(-1.0)
             .otherwise(0.0)
             .alias("signal")
        )
        
        return []
    
    return generate_signals
