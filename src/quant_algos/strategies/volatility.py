"""Volatility-based strategies."""

from typing import Callable

import polars as pl


def volatility_arbitrage(
    window: int = 20,
    threshold: float = 0.02
) -> Callable:
    """Volatility arbitrage strategy."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        close = pl.col("close")
        
        # Calculate realized volatility
        returns = close.pct_change()
        realized_vol = returns.pow(2).rolling_mean(window_size=window).sqrt()
        
        # Calculate implied volatility (placeholder)
        implied_vol = realized_vol * 0.8  # Placeholder
        
        # Spread
        spread = realized_vol - implied_vol
        
        df = data.with_columns([
            realized_vol.alias("realized_vol"),
            spread.alias("spread"),
        ])
        
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        spread_val = latest["spread"]
        
        if spread_val > threshold:
            # Realized > Implied - sell volatility
            return [{
                "symbol": latest["symbol"],
                "size": -1.0,
                "price": latest["close"],
            }]
        elif spread_val < -threshold:
            # Realized < Implied - buy volatility
            return [{
                "symbol": latest["symbol"],
                "size": 1.0,
                "price": latest["close"],
            }]
        
        return []
    
    return generate_signals
