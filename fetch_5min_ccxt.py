#!/usr/bin/env python3
"""Fetch 5-minute Bitcoin data using CCXT for the last full month."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def fetch_5min_bitcoin_ccxt(months_back: int = 1):
    """
    Fetch 5-minute Bitcoin data from Binance using CCXT.
    
    Args:
        months_back: How many months back to start (default: 1)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Get the symbol
    symbol = 'BTC/USDT'
    
    # Calculate date range
    now = datetime.now()
    end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=30*months_back)
    
    print(f"Fetching 5-minute BTC/USDT data from {start_date} to {end_date}...")
    
    all_ohlcvs = []
    
    # Fetch in batches (CCXT has limits)
    current_start = start_date
    
    while current_start < end_date:
        # Convert to milliseconds timestamp
        since = int(current_start.timestamp() * 1000)
        
        try:
            print(f"  Fetching from {current_start.strftime('%Y-%m-%d')}...")
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=1000)
            
            if ohlcvs:
                all_ohlcvs.extend(ohlcvs)
                # Update current_start to the last timestamp + 5 minutes
                last_timestamp = ohlcvs[-1][0] / 1000
                current_start = datetime.fromtimestamp(last_timestamp) + timedelta(minutes=5)
            else:
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            break
        
        # Small delay to avoid rate limiting
        import time
        time.sleep(0.5)
        
        # Safety limit
        if len(all_ohlcvs) >= 30000:  # ~21 days of 5-min data
            break
    
    if not all_ohlcvs:
        raise ValueError("No data fetched!")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    try:
        data = fetch_5min_bitcoin_ccxt(months_back=1)
        
        print(f"\nTotal rows: {len(data)}")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
        
        # Save to CSV
        output_path = OUTPUT_DIR / "bitcoin_5min_last_month.csv"
        data.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
