#!/usr/bin/env python3
"""Run validation on test set using best momentum parameters."""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quant_algos import add_technical_indicators, momentum_strategy, calculate_metrics


def run_validation(data_path: str, output_path: str = None):
    """Run momentum strategy with best parameters on validation data."""
    print("=" * 60)
    print("Validation: Momentum Strategy on Real Bitcoin Data")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading validation data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    df = add_technical_indicators(df)
    
    # Run momentum strategy with best params from hyperparameter search
    print("\nRunning momentum strategy with best parameters...")
    print("  Fast Window: 5")
    print("  Slow Window: 30")
    print("  Volume Threshold: 1.5")
    
    results = momentum_strategy(
        df.copy(),
        capital=10000.0,
        fast_window=5,
        slow_window=30,
        volume_threshold=1.5,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        max_holding_period=60,
    )
    
    metrics = calculate_metrics(results, capital=10000.0)
    
    print("\n" + "-" * 60)
    print("Validation Performance Metrics:")
    print("-" * 60)
    print(f"   Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics['total_trades']}")
    
    print(f"\n   Final Equity: ${10000 * (1 + metrics['total_return']):,.2f}")
    
    # Save results
    if output_path:
        print(f"\nSaving results to {output_path}...")
        results.to_csv(output_path, index=False)
        
        # Save metrics JSON
        import json
        output = {
            'strategy': 'momentum',
            'validation_metrics': {k: float(v) for k, v in metrics.items()},
            'final_equity': float(10000 * (1 + metrics['total_return'])),
            'data_source': data_path,
        }
        with open(Path(output_path).parent / "validation_metrics.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Validation metrics saved to {Path(output_path).parent / 'validation_metrics.json'}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run validation on test set")
    parser.add_argument("--data", default="data/btc_1h_180d.csv", help="Path to data file")
    parser.add_argument("--output", default="results/validation_results.csv", help="Path to output file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    metrics = run_validation(args.data, args.output)
