#!/usr/bin/env python3
"""Test mean reversion strategy on real Bitcoin 1-hour data (180 days)."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quant_algos import add_technical_indicators, mean_reversion_strategy, calculate_metrics


def test_mean_reversion_on_real_data(data_path: str, output_path: str = None, verbose: bool = False):
    """Test mean reversion strategy on real Bitcoin data."""
    print("=" * 60)
    print("Testing Mean Reversion Strategy on Real Bitcoin Data (1-hour, 180 days)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    df = add_technical_indicators(df)
    
    print("\n" + "-" * 60)
    print("Test 1: Basic Mean Reversion Strategy")
    print("-" * 60)
    
    # Run basic mean reversion strategy
    results1 = mean_reversion_strategy(
        df.copy(),
        capital=10000.0,
        bb_window=20,
        bb_multiplier=2.0,
        volume_threshold=1.2,
        position_size_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_period=30,
        verbose=verbose,
    )
    
    metrics1 = calculate_metrics(results1, capital=10000.0)
    
    print("\nPerformance Metrics (Basic):")
    print(f"   Total Return: {metrics1['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics1['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics1['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics1['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics1['total_trades']}")
    
    print(f"\nFinal Equity: ${10000 * (1 + metrics1['total_return']):,.2f}")
    
    # Save results
    if output_path:
        print(f"\nSaving results to {output_path}...")
        results1.to_csv(output_path, index=False)
    
    return {
        'basic': metrics1,
        'results': results1,
    }


def hyperparameter_search(data_path: str):
    """Search for optimal hyperparameters."""
    print("=" * 60)
    print("Hyperparameter Search")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_technical_indicators(df)
    
    # Grid search
    windows = [15, 20, 25, 30]
    std_multipliers = [1.8, 2.0, 2.2]
    volume_thresholds = [1.0, 1.2, 1.5]
    stop_loss_pct = 0.02
    take_profit_pct = 0.03
    max_holding_period = 30
    
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    
    results_grid = []
    
    for bbw in windows:
        for bbm in std_multipliers:
            try:
                results = mean_reversion_strategy(
                    df.copy(),
                    capital=10000.0,
                    window=bbw,
                    std_multiplier=bbm,
                    position_size_pct=0.05,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    max_holding_period=max_holding_period,
                )
                
                metrics = calculate_metrics(results, capital=10000.0)
                
                # Score: Sharpe - 0.5 * MaxDrawdown
                score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
                
                results_grid.append({
                    'window': bbw,
                    'std_multiplier': bbm,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'score': score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'window': bbw,
                        'std_multiplier': bbm,
                    }
                    best_metrics = metrics
                
                print(f"  W={bbw}, M={bbm}: Return={metrics['total_return']*100:.2f}%, "
                      f"Sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['max_drawdown']*100:.2f}%, "
                      f"Trades={metrics['total_trades']}, Score={score:.2f}")
                
            except Exception as e:
                print(f"  W={bbw}, M={bbm}: Error - {e}")
    
    print("\n" + "-" * 60)
    print("Best Parameters Found:")
    print(f"  Window: {best_params['window']}")
    print(f"  Std Multiplier: {best_params['std_multiplier']}")
    print(f"  Score: {best_score:.2f}")
    print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    
    return best_params, best_metrics, results_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mean reversion strategy on real Bitcoin data")
    parser.add_argument("--data", default="data/btc_1h_180d.csv", help="Path to data file")
    parser.add_argument("--output", default="results/mean_reversion_1h_180d_results.csv", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.search:
        best_params, best_metrics, results_grid = hyperparameter_search(args.data)
        
        # Save grid search results
        grid_df = pd.DataFrame(results_grid)
        grid_df.to_csv(Path(args.output).parent / "mean_reversion_1h_180d_grid_search.csv", index=False)
        print(f"\nGrid search results saved to {Path(args.output).parent / 'mean_reversion_1h_180d_grid_search.csv'}")
        
        # Save best params
        import json
        output = {
            'strategy': 'mean_reversion',
            'best_params': best_params,
            'best_metrics': {k: float(v) for k, v in best_metrics.items()},
            'best_score': float(best_score),
            'total_samples': len(results_grid),
        }
        with open(Path(args.output).parent / "mean_reversion_1h_180d_best_params.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Best parameters saved to {Path(args.output).parent / 'mean_reversion_1h_180d_best_params.json'}")
    else:
        results = test_mean_reversion_on_real_data(args.data, args.output, args.verbose)
