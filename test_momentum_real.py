#!/usr/bin/env python3
"""Test momentum strategy on real Bitcoin 1-hour data (180 days)."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quant_algos import add_technical_indicators, momentum_strategy, calculate_metrics, enhanced_momentum_strategy


def test_momentum_on_real_data(data_path: str, output_path: str = None, verbose: bool = False):
    """Test momentum strategy on real Bitcoin data."""
    print("=" * 60)
    print("Testing Momentum Strategy on Real Bitcoin Data (1-hour, 180 days)")
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
    print("Test 1: Basic Momentum Strategy")
    print("-" * 60)
    
    # Run basic momentum strategy
    results1 = momentum_strategy(
        df.copy(),
        capital=10000.0,
        fast_window=5,
        slow_window=20,
        volume_threshold=1.5,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        max_holding_period=60,
        verbose=verbose,
    )
    
    # Debug signals
    print("\nSignal distribution:")
    print(f"  Long signals: {(results1['signal'] == 1).sum()}")
    print(f"  Short signals: {(results1['signal'] == -1).sum()}")
    print(f"  Volume MA range: {results1['volume_ma'].min():.0f} - {results1['volume_ma'].max():.0f}")
    print(f"  Volume range: {results1['volume'].min():.0f} - {results1['volume'].max():.0f}")
    
    metrics1 = calculate_metrics(results1, capital=10000.0)
    
    print("\nPerformance Metrics (Basic):")
    print(f"   Total Return: {metrics1['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics1['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics1['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics1['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics1['total_trades']}")
    
    print(f"\nFinal Equity: ${10000 * (1 + metrics1['total_return']):,.2f}")
    
    print("\n" + "-" * 60)
    print("Test 2: Enhanced Momentum Strategy (with RSI & volatility filter)")
    print("-" * 60)
    
    # Run enhanced momentum strategy
    results2 = enhanced_momentum_strategy(
        df.copy(),
        capital=10000.0,
        fast_window=5,
        slow_window=20,
        volume_threshold=1.5,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        max_holding_period=60,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        use_rsi_filter=True,
        use_volatility_filter=True,
        verbose=verbose,
    )
    
    metrics2 = calculate_metrics(results2, capital=10000.0)
    
    print("\nPerformance Metrics (Enhanced):")
    print(f"   Total Return: {metrics2['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics2['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics2['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics2['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics2['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics2['total_trades']}")
    
    print(f"\nFinal Equity: ${10000 * (1 + metrics2['total_return']):,.2f}")
    
    # Save results
    if output_path:
        print(f"\nSaving results to {output_path}...")
        results2.to_csv(output_path, index=False)
    
    return {
        'basic': metrics1,
        'enhanced': metrics2,
        'results': results2,
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
    
    # Grid search - use basic momentum since enhanced needs pre-computed indicators
    fast_windows = [3, 5, 8, 10]
    slow_windows = [10, 20, 30]
    volume_thresholds = [1.2, 1.5, 2.0]
    stop_loss_pct = 0.015
    take_profit_pct = 0.03
    max_holding_period = 60
    
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    
    results_grid = []
    
    for fw in fast_windows:
        for sw in slow_windows:
            for vt in volume_thresholds:
                try:
                    results = momentum_strategy(
                        df.copy(),
                        capital=10000.0,
                        fast_window=fw,
                        slow_window=sw,
                        volume_threshold=vt,
                        position_size_pct=0.05,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                        max_holding_period=max_holding_period,
                    )
                    
                    metrics = calculate_metrics(results, capital=10000.0)
                    
                    # Score: Sharpe - 0.5 * MaxDrawdown
                    score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
                    
                    results_grid.append({
                        'fast_window': fw,
                        'slow_window': sw,
                        'volume_threshold': vt,
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
                            'fast_window': fw,
                            'slow_window': sw,
                            'volume_threshold': vt,
                        }
                        best_metrics = metrics
                    
                    print(f"  FW={fw}, SW={sw}, VT={vt}: Return={metrics['total_return']*100:.2f}%, "
                          f"Sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['max_drawdown']*100:.2f}%, "
                          f"Trades={metrics['total_trades']}, Score={score:.2f}")
                    
                except Exception as e:
                    print(f"  FW={fw}, SW={sw}, VT={vt}: Error - {e}")
    
    print("\n" + "-" * 60)
    print("Best Parameters Found:")
    print(f"  Fast Window: {best_params['fast_window']}")
    print(f"  Slow Window: {best_params['slow_window']}")
    print(f"  Volume Threshold: {best_params['volume_threshold']}")
    print(f"  Score: {best_score:.2f}")
    print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    
    # Save grid search results
    grid_df = pd.DataFrame(results_grid)
    grid_df.to_csv(Path(args.output).parent / "momentum_1h_180d_grid_search.csv", index=False)
    
    # Save best params
    import json
    output = {
        'strategy': 'momentum',
        'best_params': best_params,
        'best_metrics': {k: float(v) for k, v in best_metrics.items()},
        'best_score': float(best_score),
        'total_samples': len(results_grid),
    }
    with open(Path(args.output).parent / "momentum_1h_180d_best_params.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    return best_params, best_metrics, results_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test momentum strategy on real Bitcoin data")
    parser.add_argument("--data", default="data/btc_1h_180d.csv", help="Path to data file")
    parser.add_argument("--output", default="results/momentum_1h_180d_results.csv", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--search", action="store_true", help="Run hyperparameter search")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.search:
        best_params, best_metrics, results_grid = hyperparameter_search(args.data)
    else:
        results = test_momentum_on_real_data(args.data, args.output, args.verbose)
