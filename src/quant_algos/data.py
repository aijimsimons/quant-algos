"""Data loading and generation utilities for short-term trading."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta


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
    volumes = np.zeros(n_minutes)
    
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
        volumes[t] = int(1000 + 5000 * volatility_process[t])
    
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


def fetch_bitcoin_data(
    period: str = "2y",
    interval: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch real Bitcoin historical data from Yahoo Finance.
    
    Args:
        period: Data period (e.g., '1y', '2y', '5y', 'max')
        interval: Data interval (e.g., '1h', '4h', '1d')
        start_date: Start date string (YYYY-MM-DD) - overrides period
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker("BTC-USD")
    
    if start_date and end_date:
        data = ticker.history(start=start_date, end=end_date, interval=interval)
    else:
        data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data returned for BTC-USD with period={period}, interval={interval}")
    
    # Reset index to get timestamp as column
    data = data.reset_index()
    
    # Rename columns to match expected format
    # Handle both 'Date' and 'Datetime' column names
    if 'Datetime' in data.columns:
        data = data.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })
    else:
        data = data.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })
    
    # Remove timezone info for consistency
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    
    # Ensure correct column order
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    return data


def fetch_bitcoin_minute_data(days: int = 30) -> pd.DataFrame:
    """
    Fetch real Bitcoin minute-level data.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # For minute data, we can only get 7 days max from Yahoo Finance
    # So we'll fetch multiple chunks
    all_data = []
    
    # Yahoo Finance allows 7 days of 1-minute data
    chunk_days = min(days, 7)
    
    for i in range(0, days, chunk_days):
        chunk_end = end_date - timedelta(days=i)
        chunk_start = chunk_end - timedelta(days=chunk_days)
        
        try:
            data = fetch_bitcoin_data(
                start_date=chunk_start.strftime('%Y-%m-%d'),
                end_date=chunk_end.strftime('%Y-%m-%d'),
                interval='1m'
            )
            if not data.empty:
                all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not fetch data for {chunk_start} to {chunk_end}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        # Fallback to synthetic data if real data unavailable
        print("Warning: Falling back to synthetic data")
        return generate_minute_bars(n_days=days)


def fetch_bitcoin_hourly_data(days: int = 365) -> pd.DataFrame:
    """
    Fetch real Bitcoin hourly data for longer backtesting.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = fetch_bitcoin_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    return data


def fetch_bitcoin_daily_data(years: int = 3) -> pd.DataFrame:
    """
    Fetch real Bitcoin daily data for extended backtesting.
    
    Args:
        years: Number of years of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = fetch_bitcoin_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )
    
    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Z-score
    df['zscore'] = (df['close'] - df['bb_middle']) / df['bb_std']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Price momentum
    df['momentum_5'] = df['close'].pct_change(periods=5)
    df['momentum_10'] = df['close'].pct_change(periods=10)
    df['momentum_20'] = df['close'].pct_change(periods=20)
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df


def split_train_validation_test(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with OHLCV data
        train_ratio: Proportion for training
        validation_ratio: Proportion for validation
        
    Returns:
        Tuple of (train_df, validation_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, validation_df, test_df


def normalize_data(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize features to z-scores.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize (None = all numeric)
        
    Returns:
        Tuple of (normalized_df, stats_dict)
    """
    df = df.copy()
    
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            stats[col] = {'mean': mean, 'std': std}
    
    return df, stats
