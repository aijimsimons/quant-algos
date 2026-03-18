"""Volatility regime detection strategy."""

from typing import Callable

import polars as pl


def regime_detection_strategy(
    window: int = 20,
    volatility_threshold: float = 0.02
) -> Callable:
    """Volatility regime detection strategy."""
    def generate_signals(
        data: pl.DataFrame,
        positions: list,
        capital: float
    ) -> list[dict]:
        close = pl.col("close")
        
        # Calculate rolling volatility
        returns = close.pct_change()
        volatility = returns.pow(2).rolling_mean(window_size=window).sqrt()
        
        # Calculate volatility regime
        avg_vol = volatility.rolling_mean(window_size=50)
        is_high_vol = volatility > avg_vol * 1.5
        
        df = data.with_columns([
            volatility.alias("volatility"),
            avg_vol.alias("avg_volatility"),
            is_high_vol.alias("high_volatility"),
        ])
        
        latest = df.sort("datetime").tail(1).to_dicts()[0]
        vol = latest["volatility"]
        avg = latest["avg_volatility"]
        high_vol = latest["high_volatility"]
        
        # High volatility = mean reversion, Low volatility = trend following
        if high_vol:
            # Mean reversion in high vol
            if vol > volatility_threshold:
                return [{
                    "symbol": latest["symbol"],
                    "size": -0.5,
                    "price": latest["close"],
                }]
        else:
            # Trend following in low vol
            if vol < volatility_threshold * 0.5:
                return [{
                    "symbol": latest["symbol"],
                    "size": 0.5,
                    "price": latest["close"],
                }]
        
        return []
    
    return generate_signals
