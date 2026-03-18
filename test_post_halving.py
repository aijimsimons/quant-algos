#!/usr/bin/env python3
"""Test with post-halving data (April 2024 - March 2026)."""

from quant_algos import (
    fetch_bitcoin_data,
    add_technical_indicators,
    mean_reversion_strategy,
    calculate_metrics,
)

# Fetch hourly data from April 2024 to March 2026
# This covers the full post-halving cycle
data = fetch_bitcoin_data(
    start_date='2024-04-01',
    end_date='2026-03-31',
    interval='1h'
)

print(f"Rows: {len(data)}")
print(f"First: {data['timestamp'].min()}")
print(f"Last: {data['timestamp'].max()}")
print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")

# Add indicators
data = add_technical_indicators(data)

# Run strategy with best params
result = mean_reversion_strategy(
    data.copy(),
    capital=10000.0,
    window=30,
    std_multiplier=2.0,
    position_size_pct=0.05,
    stop_loss_pct=0.015,
    take_profit_pct=0.025,
    max_holding_period=30
)

metrics = calculate_metrics(result, capital=10000.0)

print("\nPerformance Metrics (Post-Halving 2024-2026):")
print(f"  Total Return: {metrics['total_return']*100:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
print(f"  Total Trades: {metrics['total_trades']}")

# Save results
import json
from pathlib import Path

output = {
    'strategy': 'mean_reversion',
    'best_params': {
        'window': 30,
        'std_multiplier': 2.0,
        'position_size_pct': 0.05,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.025,
        'max_holding_period': 30,
    },
    'best_score': metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown'],
    'best_metrics': {k: float(v) for k, v in metrics.items()},
    'data_source': 'real_bitcoin_hourly',
    'data_start': '2024-04-01',
    'data_end': '2026-03-31',
    'data_years': 'post-halving cycle',
}

with open(Path('train_results/mean_reversion_train_post_halving.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved!")
