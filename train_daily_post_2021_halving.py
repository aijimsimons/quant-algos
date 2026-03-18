#!/usr/bin/env python3
"""Train with daily Bitcoin data from post-2021 halving (2022-2026)."""

from quant_algos import (
    fetch_bitcoin_daily_data,
    add_technical_indicators,
    mean_reversion_strategy,
    calculate_metrics,
)

import pandas as pd
import numpy as np
from pathlib import Path
import json

OUTPUT_DIR = Path("train_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_training():
    """Train with 2022-2026 daily data (post-2021 halving)."""
    print("=" * 60)
    print("  DAILY TRAINING: Post-2021 Halving (2022-2026)")
    print("=" * 60)
    
    # Fetch daily data (March 2021 - March 2026)
    print("\nFetching daily Bitcoin data (2021-2026)...")
    data = fetch_bitcoin_daily_data(years=5)
    print(f"  Data: {len(data)} rows")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    # Filter to 2022-2026 (post-2021 halving)
    data = data[data['timestamp'] >= '2022-01-01'].copy()
    print(f"  After filtering (2022+): {len(data)} rows")
    
    # Add indicators
    data = add_technical_indicators(data)
    
    # Parameter grid
    params = {
        'window': [10, 20, 30],
        'std_multiplier': [1.8, 2.0, 2.2],
        'position_size_pct': [0.03, 0.05, 0.07],
        'stop_loss_pct': [0.015, 0.02],
        'take_profit_pct': [0.025, 0.03],
        'max_holding_period': [30, 60, 90],
    }
    
    # Random sampling
    param_names = list(params.keys())
    param_values = list(params.values())
    
    samples = 50
    print(f"\nTesting {samples} random parameter combinations...")
    
    results = []
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    
    for i in range(samples):
        current_params = {
            name: np.random.choice(values) 
            for name, values in zip(param_names, param_values)
        }
        
        try:
            result = mean_reversion_strategy(
                data.copy(), capital=10000.0, **current_params
            )
            
            metrics = calculate_metrics(result, capital=10000.0)
            score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
            
            results.append({
                **current_params,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'score': score,
            })
            
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
                best_metrics = metrics.copy()
                
        except Exception as e:
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{samples}", end='\r')
    
    print(f"\n  {samples}/{samples}")
    
    results_df = pd.DataFrame(results).sort_values('score', ascending=False)
    
    # Print best results
    print("\n" + "-" * 60)
    print("BEST PARAMETERS FOUND:")
    print("-" * 60)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest Score: {best_score:.4f}")
    
    print("\nPerformance Metrics (2022-2026 Daily):")
    print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    
    # Save results
    output_path = OUTPUT_DIR / "mean_reversion_train_2022_daily"
    
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        return obj
    
    json_output = {
        'strategy': 'mean_reversion',
        'best_params': convert_types(best_params),
        'best_score': float(best_score),
        'best_metrics': {k: float(v) for k, v in best_metrics.items()},
        'total_samples': len(results),
        'data_source': 'real_bitcoin_daily',
        'data_start': '2022-01-01',
        'data_end': '2026-03-18',
        'data_years': 'post-2021 halving cycle',
    }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(json_output, f, indent=2)
    
    results_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return best_params, best_score, best_metrics


if __name__ == "__main__":
    best_params, best_score, best_metrics = run_training()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
