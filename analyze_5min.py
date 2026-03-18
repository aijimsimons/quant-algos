#!/usr/bin/env python3
"""Analyze 5-minute data with different parameters."""

import pandas as pd
import numpy as np
from quant_algos import add_technical_indicators, mean_reversion_strategy, calculate_metrics


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


data = load_from_csv('data/bitcoin_5min_last_month.csv')
data = add_technical_indicators(data)

# Try different parameters
test_params = [
    {'window': 20, 'std_multiplier': 2.0, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.03},
    {'window': 30, 'std_multiplier': 2.0, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.03},
    {'window': 15, 'std_multiplier': 2.0, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.025},
    {'window': 25, 'std_multiplier': 2.2, 'stop_loss_pct': 0.015, 'take_profit_pct': 0.025},
    {'window': 20, 'std_multiplier': 2.0, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.04},
    {'window': 25, 'std_multiplier': 2.0, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.04},
]

print("Testing different parameters on 5-minute data:\n")

for p in test_params:
    result = mean_reversion_strategy(
        data.copy(), 
        capital=10000.0, 
        **p, 
        position_size_pct=0.05, 
        max_holding_period=60
    )
    metrics = calculate_metrics(result, capital=10000.0)
    
    score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
    
    print(f'Window={p["window"]}, SD={p["std_multiplier"]}, SL={p["stop_loss_pct"]*100}%, TP={p["take_profit_pct"]*100}%')
    print(f'  Return: {metrics["total_return"]*100:.2f}%, Sharpe: {metrics["sharpe_ratio"]:.2f}')
    print(f'  Max DD: {metrics["max_drawdown"]*100:.2f}%, Win: {metrics["win_rate"]*100:.1f}%, Trades: {metrics["total_trades"]}')
    print(f'  Score: {score:.4f}')
    print()
