"""Momentum trading strategy implementation."""

import pandas as pd
import numpy as np


def momentum_strategy(
    data: pd.DataFrame,
    capital: float = 10000.0,
    momentum_window: int = 20,
    position_size_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Simple momentum strategy based on price trend.
    
    Args:
        data: DataFrame with 'timestamp', 'price', 'volume' columns
        capital: Initial capital in USD
        momentum_window: Lookback period for momentum calculation
        position_size_pct: Percentage of capital to risk per trade
    
    Returns:
        DataFrame with strategy signals and returns
    """
    df = data.copy()
    
    # Calculate momentum (rate of change)
    df['momentum'] = df['price'].pct_change(periods=momentum_window)
    
    # Generate signals: long if momentum > 0, flat otherwise
    df['signal'] = np.where(df['momentum'] > 0, 1, 0)
    
    # Calculate position size in units
    df['position_value'] = capital * position_size_pct
    df['size'] = df['position_value'] / df['price']
    
    # Calculate returns
    df['returns'] = df['price'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate strategy performance metrics."""
    returns = df['strategy_returns'].dropna()
    
    metrics = {
        'total_return': df['cumulative_returns'].iloc[-1] - 1,
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(365),  # annualized
        'max_drawdown': (df['cumulative_returns'].cummax() - df['cumulative_returns']).max(),
        'win_rate': (returns > 0).mean(),
    }
    
    return metrics
