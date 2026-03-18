#!/usr/bin/env python3
"""
Train-only backtesting script for Bitcoin trading strategies.

This script:
1. Fetches real Bitcoin data (hourly or daily resolution)
2. Optimizes strategy hyperparameters on TRAIN data only
3. Repeatedly improves and tests on train data
4. No validation - user gives approval before running validation

Usage:
    python train_only.py --strategy mean_reversion
    python train_only.py --strategy momentum
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

from quant_algos import (
    generate_minute_bars,
    add_technical_indicators,
    mean_reversion_strategy,
    momentum_strategy,
    calculate_metrics,
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "train_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def optimize_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    strategy_func,
    param_grid: dict,
    capital: float = 10000.0,
    scoring: str = 'combined',
    n_iterations: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Multi-iteration hyperparameter optimization.
    
    After each iteration, we:
    1. Record best results
    2. Adjust search space around best parameters
    3. Continue optimization
    
    Args:
        data: Training data (with or without indicators)
        strategy_name: Name of strategy
        strategy_func: Strategy function
        param_grid: Initial parameter grid
        capital: Initial capital
        scoring: Scoring method ('sharpe', 'return', 'combined')
        n_iterations: Number of optimization iterations
        verbose: Print progress
        
    Returns:
        Dictionary with best parameters and metrics
    """
    # Add indicators if not already present
    if 'sma' not in data.columns and 'rsi' not in data.columns:
        data = add_technical_indicators(data.copy())
    
    best_results = {
        'best_params': None,
        'best_score': -float('inf'),
        'best_metrics': None,
        'all_results': [],
        'iterations': [],
    }
    
    current_param_grid = param_grid.copy()
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        
        param_names = list(current_param_grid.keys())
        param_values = list(current_param_grid.values())
        total_combinations = np.prod([len(v) for v in param_values])
        
        if verbose:
            print(f"Parameter grid: {total_combinations} combinations")
        
        best_iteration_score = -float('inf')
        best_iteration_params = None
        best_iteration_metrics = None
        
        combo_count = 0
        for combo in np.ndindex(*[len(v) for v in param_values]):
            combo_count += 1
            if verbose and combo_count % 100 == 0:
                print(f"  {combo_count}/{total_combinations}", end='\r')
            
            current_params = {
                param_names[i]: param_values[i][combo[i]] 
                for i in range(len(param_names))
            }
            
            try:
                result = strategy_func(
                    data.copy(),
                    capital=capital,
                    **current_params,
                    verbose=False,
                )
                metrics = calculate_metrics(result, capital=capital)
                
                # Score based on specified criterion
                if scoring == 'sharpe':
                    score = metrics['sharpe_ratio']
                elif scoring == 'return':
                    score = metrics['total_return']
                elif scoring == 'combined':
                    # Combined: Sharpe - 0.5 * drawdown
                    score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
                else:
                    score = metrics['sharpe_ratio']
                
                result_entry = {
                    **current_params,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'score': score,
                    'iteration': iteration + 1,
                }
                best_results['all_results'].append(result_entry)
                
                if score > best_iteration_score:
                    best_iteration_score = score
                    best_iteration_params = current_params.copy()
                    best_iteration_metrics = metrics.copy()
                    
            except Exception as e:
                continue
        
        if verbose:
            print(f"\n  Best in iteration {iteration + 1}:")
            if best_iteration_params:
                for key, value in best_iteration_params.items():
                    print(f"    {key}: {value}")
                print(f"    Score: {best_iteration_score:.4f}")
                print(f"    Return: {best_iteration_metrics['total_return']*100:.2f}%")
                print(f"    Sharpe: {best_iteration_metrics['sharpe_ratio']:.2f}")
                print(f"    Trades: {best_iteration_metrics['total_trades']}")
        
        best_results['iterations'].append({
            'iteration': iteration + 1,
            'best_score': best_iteration_score,
            'best_params': best_iteration_params,
        })
        
        # Update best overall
        if best_iteration_score > best_results['best_score']:
            best_results['best_score'] = best_iteration_score
            best_results['best_params'] = best_iteration_params
            best_results['best_metrics'] = best_iteration_metrics
        
        # For next iteration, narrow search around best params
        if iteration < n_iterations - 1 and best_iteration_params:
            current_param_grid = narrow_search_space(
                current_param_grid, 
                best_iteration_params,
                reduction_factor=0.5
            )
    
    return best_results


def narrow_search_space(param_grid: dict, best_params: dict, reduction_factor: float = 0.5) -> dict:
    """
    Narrow parameter search space around best parameters.
    
    Args:
        param_grid: Current parameter grid
        best_params: Best parameters found
        reduction_factor: How much to reduce the range (0-1)
        
    Returns:
        Narrowed parameter grid
    """
    narrowed = {}
    
    for param, values in param_grid.items():
        if param in best_params:
            best_val = best_params[param]
            
            if isinstance(values, list) and len(values) > 1:
                # Find the position of best value
                try:
                    idx = values.index(best_val)
                    
                    # Keep best value and 1 neighbor on each side
                    start = max(0, idx - 1)
                    end = min(len(values), idx + 2)
                    narrowed[param] = values[start:end]
                except ValueError:
                    narrowed[param] = values
            else:
                narrowed[param] = values
        else:
            narrowed[param] = values
    
    return narrowed


def save_results(best_results: dict, strategy_name: str, filename: str = None):
    """Save optimization results."""
    if filename is None:
        filename = f"{strategy_name}_train_results"
    
    output_path = OUTPUT_DIR / filename
    
    # Save as JSON
    json_output = {
        'strategy': strategy_name,
        'best_params': best_results['best_params'],
        'best_score': best_results['best_score'],
        'best_metrics': best_results['best_metrics'],
        'iterations': best_results['iterations'],
        'all_results_count': len(best_results['all_results']),
    }
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        return obj
    
    json_output = convert_types(json_output)
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # Save all results as CSV
    if best_results['all_results']:
        all_results_df = pd.DataFrame(best_results['all_results'])
        all_results_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"\n  Results saved to: {output_path}")


def print_strategy_summary(best_results: dict, strategy_name: str):
    """Print a summary of the optimization results."""
    print_section(f"{strategy_name.upper()} - FINAL RESULTS")
    
    if best_results['best_params']:
        print(f"\nBest Parameters:")
        for key, value in best_results['best_params'].items():
            print(f"  {key}: {value}")
        
        print(f"\nBest Score: {best_results['best_score']:.4f}")
        
        metrics = best_results['best_metrics']
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        
        print(f"\nOptimization Iterations:")
        for iter_result in best_results['iterations']:
            print(f"  Iteration {iter_result['iteration']}: "
                  f"Score={iter_result['best_score']:.4f}")
        
        print("\n" + "-" * 50)
        print("TRAINING COMPLETE")
        print("-" * 50)
        print("\nNext steps:")
        print("1. Review the results above")
        print("2. If satisfied, ask user for validation approval")
        print("3. If not satisfied, adjust parameters and re-run")
    else:
        print("  No valid parameters found!")
        print("  Check data quality and parameter grid.")


def main():
    parser = argparse.ArgumentParser(description='Train-only backtesting')
    parser.add_argument('--strategy', '-s', required=True, 
                        choices=['mean_reversion', 'momentum'],
                        help='Strategy to train')
    parser.add_argument('--iterations', '-i', type=int, default=5,
                        help='Number of optimization iterations')
    parser.add_argument('--data-days', '-d', type=int, default=30,
                        help='Number of days of data to use')
    parser.add_argument('--capital', '-c', type=float, default=10000.0,
                        help='Initial capital')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"  {args.strategy.upper()} TRAIN-ONLY BACKTESTING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Data: {args.data_days} days")
    print(f"  Iterations: {args.iterations}")
    print(f"  Capital: ${args.capital:,.2f}")
    
    # Generate training data
    print_section("1. GENERATING TRAINING DATA")
    data = generate_minute_bars(n_days=args.data_days)
    print(f"  Generated {len(data)} rows")
    print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Define parameter grids
    if args.strategy == 'mean_reversion':
        param_grid = {
            'window': [10, 15, 20, 25, 30],
            'std_multiplier': [1.5, 1.75, 2.0, 2.25, 2.5],
            'position_size_pct': [0.03, 0.05, 0.07, 0.10],
            'stop_loss_pct': [0.01, 0.012, 0.015, 0.02],
            'take_profit_pct': [0.02, 0.025, 0.03, 0.035],
            'max_holding_period': [30, 45, 60, 90, 120],
        }
        strategy_func = mean_reversion_strategy
    else:  # momentum
        param_grid = {
            'fast_window': [3, 5, 7, 10],
            'slow_window': [10, 15, 20, 25, 30],
            'volume_threshold': [1.2, 1.5, 2.0, 2.5],
            'position_size_pct': [0.03, 0.05, 0.07, 0.10],
            'stop_loss_pct': [0.01, 0.012, 0.015, 0.02],
            'take_profit_pct': [0.02, 0.025, 0.03, 0.04],
            'max_holding_period': [30, 45, 60, 90, 120],
        }
        strategy_func = momentum_strategy
    
    # Run optimization
    print_section("2. HYPERPARAMETER OPTIMIZATION")
    best_results = optimize_strategy(
        data.copy(),
        args.strategy,
        strategy_func,
        param_grid,
        capital=args.capital,
        scoring='combined',
        n_iterations=args.iterations,
        verbose=args.verbose,
    )
    
    # Save and print results
    save_results(best_results, args.strategy)
    print_strategy_summary(best_results, args.strategy)


if __name__ == "__main__":
    main()
