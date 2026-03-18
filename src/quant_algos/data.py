"""Data loading and generation utilities for short-term trading."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


def generate_minute_bars(
    n_days: int = 30,
    start_price: float = 70000.0,
    volatility: float = 0.005,  # 0.5% per minute - more realistic for short-term
    drift: float = 0.00005,  # slight upward drift
    minutes_per_day: int = 1440,  # 24 hours
) -> pd.DataFrame:
    """
    Generate realistic minute-by-minute price data with volatility clustering.
    
    Args:
        n_days: Number of days of data
        start_price: Starting price
        volatility: Base volatility per minute (higher for short-term trading)
        drift: Base drift per minute
        minutes_per_day: Minutes per trading day
        
    Returns:
        DataFrame with timestamp, open, high, low, close, volume
    """
    np.random.seed(42)
    
    n_minutes = n_days * minutes_per_day
    
    # Initialize arrays
    prices = np.zeros(n_minutes)
    returns = np.zeros(n_minutes)
    volatility_process = np.zeros(n_minutes)
    
    # Start price
    prices[0] = start_price
    volatility_process[0] = volatility
    
    # Generate price path with volatility clustering (GARCH-like)
    for t in range(1, n_minutes):
        # Volatility clustering - high vol today means high vol tomorrow
        volatility_process[t] = (
            0.1 * volatility + 
            0.85 * volatility_process[t-1] + 
            0.05 * abs(returns[t-1])
        )
        volatility_process[t] = max(volatility_process[t], volatility * 0.3)
        volatility_process[t] = min(volatility_process[t], volatility * 4)
        
        # Random shock with time-varying volatility
        shock = np.random.normal(0, 1) * volatility_process[t]
        
        # Return = drift + shock
        returns[t] = drift + shock
        
        # Price = previous price * (1 + return)
        prices[t] = prices[t-1] * (1 + returns[t])
        
        # Keep price in reasonable range
        prices[t] = max(prices[t], start_price * 0.3)
        prices[t] = min(prices[t], start_price * 3.0)
    
    # Create OHLC from close prices
    opens = np.zeros(n_minutes)
    highs = np.zeros(n_minutes)
    lows = np.zeros(n_minutes)
    closes = np.zeros(n_minutes)
    
    for t in range(n_minutes):
        if t == 0:
            opens[t] = start_price
            highs[t] = start_price * 1.001
            lows[t] = start_price * 0.999
            closes[t] = start_price * 1.0005
        else:
            close = prices[t]
            open_price = prices[t-1]
            change = close - open_price
            
            if change >= 0:
                # Bullish bar
                opens[t] = open_price
                closes[t] = close
                highs[t] = max(open_price, close) * (1 + np.random.uniform(0, 0.0005))
                lows[t] = min(open_price, close) * (1 - np.random.uniform(0, 0.0005))
            else:
                # Bearish bar
                opens[t] = open_price
                closes[t] = close
                highs[t] = max(open_price, close) * (1 + np.random.uniform(0, 0.0005))
                lows[t] = min(open_price, close) * (1 - np.random.uniform(0, 0.0005))
    
    # Volume with volatility correlation
    volumes = (1000 + 5000 * volatility_process).astype(int)
    
    # Create timestamps
    timestamps = pd.date_range(
        start='2024-01-01 00:00:00',
        periods=n_minutes,
        freq='1min'
    )
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
    
    return df


def generate_tick_data(
    n_days: int = 30,
    start_price: float = 70000.0,
    ticks_per_minute: int = 10,
) -> pd.DataFrame:
    """
    Generate tick-by-tick data for high-frequency strategies.
    
    Args:
        n_days: Number of days
        start_price: Starting price
        ticks_per_minute: Average ticks per minute
        
    Returns:
        DataFrame with timestamp, price, volume (tick data)
    """
    np.random.seed(42)
    
    n_ticks = n_days * ticks_per_minute * 1440
    
    timestamps = []
    prices = []
    volumes = []
    
    current_price = start_price
    
    for _ in range(n_ticks):
        # Add timestamp
        tick_time = pd.Timestamp('2024-01-01 00:00:00') + pd.Timedelta(
            microseconds=np.random.randint(0, n_ticks * 60_000_000)
        )
        timestamps.append(tick_time)
        
        # Random walk with microstructure
        price_move = np.random.normal(0, 5)  # $5 average tick move
        current_price += price_move
        prices.append(max(current_price, 1000))
        
        # Tick volume
        volumes.append(np.random.randint(1, 10))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
    })
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def resample_to_bars(df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
    """
    Resample tick data to OHLC bars.
    
    Args:
        df: DataFrame with timestamp, price, volume
        freq: Resampling frequency (e.g., '5T' for 5 minutes)
        
    Returns:
        OHLCV DataFrame
    """
    df = df.set_index('timestamp')
    
    ohlc = df['price'].resample(freq).ohlc()
    volume = df['volume'].resample(freq).sum()
    
    result = pd.concat([ohlc, volume], axis=1)
    result.columns = ['open', 'high', 'low', 'close', 'volume']
    result = result.reset_index()
    
    return result.dropna()
