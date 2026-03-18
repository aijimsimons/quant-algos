#!/usr/bin/env python3
"""Training with REAL Bitcoin historical data."""

from quant_algos import (
    fetch_bitcoin_hourly_data,
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


def run_training_real_data(
    strategy: str = 'mean_reversion',
    years: int = 1,
    samples: int = 30,
):
    """
    Fast training with real Bitcoin data.
    
    Args:
        strategy: 'mean_reversion' or 'momentum'
        years: Years of historical data to use
        samples: Number of parameter combinations to try
    """
    print("=" * 60)
    print(f"  REAL DATA TRAINING: {strategy.upper()}")
    print("=" * 60)
    
    # Load REAL Bitcoin hourly data
    print(f"\nFetching {years} year(s) of Bitcoin hourly data...")
    data = fetch_bitcoin_hourly_data(days=years * 365)
    print(f"  Data: {len(data)} rows")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    # Add indicators
    data = add_technical_indicators(data)
    
    # Parameter grid
    if strategy == 'mean_reversion':
        params = {
            'window': [15, 20, 25, 30],
            'std_multiplier': [1.8, 2.0, 2.2],
            'position_size_pct': [0.03, 0.05, 0.07],
            'stop_loss_pct': [0.015, 0.02],
            'take_profit_pct': [0.025, 0.03],
            'max_holding_period': [30, 60, 90],
        }
    else:  # momentum
        params = {
            'fast_window': [5, 10],
            'slow_window': [15, 20],
            'volume_threshold': [1.5, 2.0],
            'position_size_pct': [0.05, 0.07],
            'stop_loss_pct': [0.015, 0.02],
            'take_profit_pct': [0.025, 0.035],
            'max_holding_period': [45, 60, 90],
        }
    
    # Random sampling
    param_names = list(params.keys())
    param_values = list(params.values())
    
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
            if strategy == 'mean_reversion':
                result = mean_reversion_strategy(
                    data.copy(), capital=10000.0, **current_params
                )
            else:
                result = momentum_strategy(
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
    
    print("\nPerformance Metrics (REAL DATA):")
    print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    
    # Save results
    output_path = OUTPUT_DIR / f"{strategy}_train_real_{years}y"
    
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
        'strategy': strategy,
        'best_params': convert_types(best_params),
        'best_score': float(best_score),
        'best_metrics': {k: float(v) for k, v in best_metrics.items()},
        'total_samples': len(results),
        'data_source': 'real_bitcoin_hourly',
        'data_years': years,
        'data_rows': len(data),
    }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(json_output, f, indent=2)
    
    results_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return best_params, best_score, best_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', '-s', default='mean_reversion',
                       choices=['mean_reversion'])
    parser.add_argument('--years', '-y', type=int, default=1)
    parser.add_argument('--samples', '-n', type=int, default=30)
    
    args = parser.parse_args()
    
    best_params, best_score, best_metrics = run_training_real_data(
        strategy=args.strategy,
        years=args.years,
        samples=args.samples,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
