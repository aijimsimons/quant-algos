#!/usr/bin/env python3
"""Test what data is available from Binance via CCXT."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_with_retry(exchange, symbol, timeframe, since, limit=1000):
    """Fetch OHLCV with retry logic."""
    for attempt in range(3):
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            return ohlcvs
        except ccxt.RateLimitExceeded:
            print(f"  Rate limited, waiting...")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            return None
    return None


def fetch_bitcoin_data_ccxt(timeframe='5m', days=30):
    """Fetch Bitcoin data from Binance."""
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"Fetching {timeframe} Bitcoin data for last {days} days...")
    
    all_ohlcvs = []
    current_since = int(start_time.timestamp() * 1000)
    end_since = int(end_time.timestamp() * 1000)
    
    while current_since < end_since:
        ohlcvs = fetch_with_retry(exchange, symbol, timeframe, current_since)
        
        if ohlcvs is None or len(ohlcvs) == 0:
            break
            
        all_ohlcvs.extend(ohlcvs)
        
        # Update since to the last timestamp + timeframe
        last_timestamp = ohlcvs[-1][0]
        current_since = last_timestamp + 60000  # Add 1 minute to avoid overlap
        
        # Rate limit
        time.sleep(0.5)
        
        # Safety limit
        if len(all_ohlcvs) >= 50000:
            break
        
        # Progress
        if len(all_ohlcvs) % 10000 == 0:
            print(f"  Fetched {len(all_ohlcvs)} rows...")
    
    if not all_ohlcvs:
        raise ValueError("No data fetched!")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Test different timeframes
    timeframes = [
        ('1m', 7),    # 1-minute for 7 days
        ('5m', 30),   # 5-minute for 30 days  
        ('15m', 90),  # 15-minute for 90 days
        ('1h', 180),  # 1-hour for 180 days
    ]
    
    print("Testing Binance data availability:\n")
    
    for timeframe, days in timeframes:
        try:
            data = fetch_bitcoin_data_ccxt(timeframe=timeframe, days=days)
            print(f"\n{timeframe} data for {days} days:")
            print(f"  Rows: {len(data)}")
            print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"  Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
        except Exception as e:
            print(f"\n{timeframe} data: ERROR - {e}")
