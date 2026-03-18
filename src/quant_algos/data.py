"""Data loading utilities for backtesting."""

import pandas as pd
from pathlib import Path


def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Load price data from CSV file.
    
    Expected columns: timestamp, price, volume
    """
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'price', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def generate_sample_data(n_days: int = 365, start_price: float = 70000.0) -> pd.DataFrame:
    """
    Generate sample price data for testing.
    
    Args:
        n_days: Number of days of data to generate
        start_price: Starting price
        
    Returns:
        DataFrame with timestamp, price, volume columns
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Generate random walk with drift
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = start_price * (1 + pd.Series(returns)).cumprod()
    
    # Generate volume
    volume = np.random.randint(100, 1000, n_days)
    
    # Create timestamp
    timestamps = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices.values,
        'volume': volume,
    })
    
    return df
