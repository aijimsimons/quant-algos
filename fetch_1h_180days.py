#!/usr/bin/env python3
"""Fetch 1-hour Bitcoin data for 180 days (post-halving)."""

import pandas as pd
import sys
sys.path.insert(0, '/Users/xingjianliu/jim/quant-algos/src')

from quant_algos import add_technical_indicators, momentum_strategy, calculate_metrics


def fetch_bitcoin_data_ccxt(timeframe='1h', days=180):
    """Fetch Bitcoin data from Binance."""
    import ccxt
    import time
    
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"Fetching {timeframe} Bitcoin data for last {days} days...")
    
    all_ohlcvs = []
    current_since = int(start_time.timestamp() * 1000)
    end_since = int(end_time.timestamp() * 1000)
    
    while current_since < end_since:
        for attempt in range(3):
            try:
                ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
                if ohlcvs:
                    all_ohlcvs.extend(ohlcvs)
                    last_timestamp = ohlcvs[-1][0]
                    current_since = last_timestamp + 60000
                    break
                else:
                    break
            except Exception as e:
                if attempt == 2:
                    break
                time.sleep(1)
        
        time.sleep(0.5)
        
        if len(all_ohlcvs) >= 10000:
            break
    
    if not all_ohlcvs:
        raise ValueError("No data fetched!")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


from datetime import datetime, timedelta

data = fetch_bitcoin_data_ccxt(timeframe='1h', days=180)

print(f"\nTotal rows: {len(data)}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

# Add indicators
data = add_technical_indicators(data)

# Run momentum strategy with best params from 5-min
print("\nRunning momentum strategy with 1-hour data...")
result = momentum_strategy(
    data.copy(), 
    capital=10000.0,
    fast_window=5,
    slow_window=10,
    volume_threshold=1.2,
    position_size_pct=0.03,
    stop_loss_pct=0.025,
    take_profit_pct=0.035,
    max_holding_period=90,
)

metrics = calculate_metrics(result, capital=10000.0)

print("\nPerformance Metrics (1-Hour 180 Days):")
print(f"  Total Return: {metrics['total_return']*100:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
print(f"  Total Trades: {metrics['total_trades']}")

# Save results
import json
from pathlib import Path

output = {
    'strategy': 'momentum',
    'best_params': {
        'fast_window': 5,
        'slow_window': 10,
        'volume_threshold': 1.2,
        'position_size_pct': 0.03,
        'stop_loss_pct': 0.025,
        'take_profit_pct': 0.035,
        'max_holding_period': 90,
    },
    'best_score': metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown'],
    'best_metrics': {k: float(v) for k, v in metrics.items()},
    'total_samples': 1,
    'data_source': 'real_bitcoin_1h_ccxt_180days',
    'data_start': str(data['timestamp'].min()),
    'data_end': str(data['timestamp'].max()),
    'data_rows': len(data),
}

with open(Path('train_results/momentum_train_1h_180days.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved!")
