#!/usr/bin/env python3
"""Train on 5-minute Bitcoin data from last full month."""

from quant_algos import (
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


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def run_training():
    """Train with 5-minute data from last month."""
    print("=" * 60)
    print("  5-MINUTE TRAINING: Last Full Month")
    print("=" * 60)
    
    # Load 5-minute data
    print("\nLoading 5-minute Bitcoin data...")
    data = load_from_csv("data/bitcoin_5min_last_month.csv")
    print(f"  Data: {len(data)} rows")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    # Add indicators
    data = add_technical_indicators(data)
    
    # Parameter grid (tighter for 5-minute data)
    params = {
        'window': [10, 20, 30],
        'std_multiplier': [1.8, 2.0, 2.2],
        'position_size_pct': [0.02, 0.05, 0.07],
        'stop_loss_pct': [0.01, 0.015, 0.02],
        'take_profit_pct': [0.015, 0.025, 0.03],
        'max_holding_period': [10, 30, 60],  # Shorter for 5-min
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
    
    print("\nPerformance Metrics (5-Min Last Month):")
    print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    
    # Save results
    output_path = OUTPUT_DIR / "mean_reversion_train_5min_last_month"
    
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
        'data_source': 'real_bitcoin_5min_ccxt',
        'data_start': str(data['timestamp'].min()),
        'data_end': str(data['timestamp'].max()),
        'data_rows': len(data),
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
