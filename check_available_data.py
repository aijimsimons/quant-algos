#!/usr/bin/env python3
"""Check what Bitcoin data is available from Yahoo Finance."""

import yfinance as yf

ticker = yf.Ticker("BTC-USD")

# Test different periods and intervals
test_cases = [
    ("1d", "2y", "Daily 2 years"),
    ("1h", "1mo", "Hourly 1 month"),
    ("5m", "1mo", "5min 1 month"),
    ("1d", "5y", "Daily 5 years"),
]

print("Testing available data from Yahoo Finance:\n")

for interval, period, desc in test_cases:
    try:
        data = ticker.history(period=period, interval=interval)
        if not data.empty:
            print(f"{desc} ({interval}, {period}):")
            print(f"  Rows: {len(data)}")
            print(f"  Range: {data.index.min()} to {data.index.max()}")
            print(f"  Close range: ${data['Close'].min():,.0f} - ${data['Close'].max():,.0f}")
        else:
            print(f"{desc}: Empty")
    except Exception as e:
        print(f"{desc}: Error - {str(e)[:80]}")
    print()
