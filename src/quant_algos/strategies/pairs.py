"""Pairs trading strategy."""

from typing import Callable

import polars as pl


def pairs_strategy(
    symbol1: str,
    symbol2: str,
    z_threshold: float = 2.0,
    window: int = 20
) -> Callable:
    """Pair trading strategy using z-score of spread."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        # Filter for the two symbols
        df1 = data.filter(pl.col("symbol") == symbol1)
        df2 = data.filter(pl.col("symbol") == symbol2)
        
        if len(df1) == 0 or len(df2) == 0:
            return []
        
        # Get latest prices
        latest1 = df1.sort("datetime").tail(1).to_dicts()[0]
        latest2 = df2.sort("datetime").tail(1).to_dicts()[0]
        
        price1 = latest1["close"]
        price2 = latest2["close"]
        
        # Calculate spread (price1 - price2)
        # For now, just return a placeholder
        # TODO: Implement actual pairs trading logic
        return []
    
    return generate_signals
