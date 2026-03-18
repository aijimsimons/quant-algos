#!/usr/bin/env python3
"""Fetch 5-minute Bitcoin data from April 2024 (post-halving) to March 2025."""

import yfinance as yf
import pandas as pd

# Halving was April 2024, so test the post-halving cycle
# We need 5-minute data for Polymarket binary options

print("Fetching 5-minute Bitcoin data (April 2024 - March 2025)...")

ticker = yf.Ticker("BTC-USD")

# Try to get 5-minute data for the post-halving period
# Yahoo Finance typically allows 60 days of 5-minute data
# We'll need to fetch in chunks

from datetime import datetime, timedelta

start_date = datetime(2024, 4, 1)
end_date = datetime(2025, 3, 31)

# Fetch in 30-day chunks (Yahoo limit for 5-minute data)
all_data = []
current_start = start_date

while current_start < end_date:
    chunk_end = min(current_start + timedelta(days=30), end_date)
    
    try:
        print(f"Fetching {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
        data = ticker.history(
            start=current_start.strftime('%Y-%m-%d'),
            end=chunk_end.strftime('%Y-%m-%d'),
            interval='5m'
        )
        
        if not data.empty:
            all_data.append(data)
            print(f"  Got {len(data)} rows")
        else:
            print(f"  No data for this period")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    current_start = chunk_end

if all_data:
    combined = pd.concat(all_data)
    
    # Reset index and rename columns
    combined = combined.reset_index()
    combined = combined.rename(columns={
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    })
    
    # Remove timezone
    combined['timestamp'] = pd.to_datetime(combined['timestamp']).dt.tz_localize(None)
    
    # Select required columns
    combined = combined[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sort by timestamp
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal: {len(combined)} rows")
    print(f"First: {combined['timestamp'].min()}")
    print(f"Last: {combined['timestamp'].max()}")
    print(f"Price range: ${combined['close'].min():,.0f} - ${combined['close'].max():,.0f}")
    
    # Save to CSV
    combined.to_csv('data/bitcoin_5min_post_halving.csv', index=False)
    print(f"\nSaved to data/bitcoin_5min_post_halving.csv")
else:
    print("No data fetched!")
