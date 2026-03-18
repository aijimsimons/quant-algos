#!/usr/bin/env python3
"""
Real data backtesting script for Bitcoin trading strategies.

This script:
1. Fetches real Bitcoin data (hourly resolution)
2. Splits data into TRAIN (60%), VALIDATION (20%), TEST (20%)
3. Optimizes strategy hyperparameters on TRAIN data
4. Validates on VALIDATION data (single run, no further tuning)
5. Reports final results on TEST data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Import our strategy modules
from quant_algos import (
    fetch_bitcoin_hourly_data,
    fetch_bitcoin_daily_data,
    add_technical_indicators,
    split_train_validation_test,
    mean_reversion_strategy,
    momentum_strategy,
    calculate_metrics,
)

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_train_validation_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_name: str,
    strategy_func,
    hyperparams: dict,
    capital: float = 10000.0,
    verbose: bool = False,
) -> dict:
    """
    Run backtest with proper train/validation/test split.
    
    Returns metrics for each dataset split.
    """
    print(f"\nRunning {strategy_name}...")
    
    # Add technical indicators to all datasets
    train_with_indicators = add_technical_indicators(train_df)
    val_with_indicators = add_technical_indicators(val_df)
    test_with_indicators = add_technical_indicators(test_df)
    
    results = {
        'strategy': strategy_name,
        'hyperparams': hyperparams,
        'train': None,
        'validation': None,
        'test': None,
    }
    
    # Train
    print(f"\n  TRAIN set ({len(train_df)} rows)...")
    try:
        train_result = strategy_func(
            train_with_indicators,
            capital=capital,
            **hyperparams,
            verbose=False,
        )
        train_metrics = calculate_metrics(train_result, capital=capital)
        results['train'] = {
            'df': train_result,
            'metrics': train_metrics,
        }
        print(f"    Return: {train_metrics['total_return']*100:.2f}%")
        print(f"    Sharpe: {train_metrics['sharpe_ratio']:.2f}")
        print(f"    Trades: {train_metrics['total_trades']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    
    # Validation - THIS IS THE CRITICAL STEP
    # No further tuning allowed after this!
    print(f"\n  VALIDATION set ({len(val_df)} rows)...")
    try:
        val_result = strategy_func(
            val_with_indicators,
            capital=capital,
            **hyperparams,
            verbose=False,
        )
        val_metrics = calculate_metrics(val_result, capital=capital)
        results['validation'] = {
            'df': val_result,
            'metrics': val_metrics,
        }
        print(f"    Return: {val_metrics['total_return']*100:.2f}%")
        print(f"    Sharpe: {val_metrics['sharpe_ratio']:.2f}")
        print(f"    Trades: {val_metrics['total_trades']}")
        
        # Validation check - strategy must have minimum performance
        if val_metrics['total_trades'] < 2:
            print("    WARNING: Not enough trades for validation!")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    
    # Test - final evaluation
    print(f"\n  TEST set ({len(test_df)} rows)...")
    try:
        test_result = strategy_func(
            test_with_indicators,
            capital=capital,
            **hyperparams,
            verbose=False,
        )
        test_metrics = calculate_metrics(test_result, capital=capital)
        results['test'] = {
            'df': test_result,
            'metrics': test_metrics,
        }
        print(f"    Return: {test_metrics['total_return']*100:.2f}%")
        print(f"    Sharpe: {test_metrics['sharpe_ratio']:.2f}")
        print(f"    Trades: {test_metrics['total_trades']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    
    return results


def optimize_hyperparameters(
    train_df: pd.DataFrame,
    strategy_name: str,
    strategy_func,
    param_grid: dict,
    capital: float = 10000.0,
    scoring: str = 'sharpe',  # 'sharpe', 'return', 'combined'
) -> dict:
    """
    Grid search over hyperparameter space using TRAIN data only.
    
    Returns best parameters found.
    """
    print(f"\nOptimizing {strategy_name} hyperparameters...")
    print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    # Add technical indicators
    train_with_indicators = add_technical_indicators(train_df)
    
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    all_results = []
    
    # Convert param_grid to list of all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    total_combinations = np.prod([len(v) for v in param_values])
    print(f"\nTesting {total_combinations} combinations...")
    
    combo_count = 0
    for combo in np.ndindex(*[len(v) for v in param_values]):
        combo_count += 1
        if combo_count % 50 == 0:
            print(f"  Progress: {combo_count}/{total_combinations}", end='\r')
        
        current_params = {param_names[i]: param_values[i][combo[i]] for i in range(len(param_names))}
        
        try:
            result = strategy_func(
                train_with_indicators.copy(),
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
                # Combined score: Sharpe with penalty for max drawdown
                score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
            else:
                score = metrics['sharpe_ratio']
            
            all_results.append({
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
            # Skip failing combinations
            continue
    
    if best_params is None:
        print(f"\n  No valid parameters found!")
        return {
            'best_params': None,
            'best_score': None,
            'best_metrics': None,
            'all_results': pd.DataFrame(all_results) if all_results else pd.DataFrame(),
        }
    
    print(f"\n  Best parameters found:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    print(f"    Best score ({scoring}): {best_score:.4f}")
    print(f"    Train metrics:")
    print(f"      Return: {best_metrics['total_return']*100:.2f}%")
    print(f"      Sharpe: {best_metrics['sharpe_ratio']:.2f}")
    print(f"      Max DD: {best_metrics['max_drawdown']*100:.2f}%")
    print(f"      Trades: {best_metrics['total_trades']}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'all_results': pd.DataFrame(all_results),
    }


def save_results(results: dict, filename: str):
    """Save backtest results to JSON and CSV files."""
    output_path = OUTPUT_DIR / filename
    
    # Save metrics as JSON
    metrics_output = {
        'strategy': results['strategy'],
        'hyperparams': results['hyperparams'],
        'train': results['train']['metrics'] if results['train'] else None,
        'validation': results['validation']['metrics'] if results['validation'] else None,
        'test': results['test']['metrics'] if results['test'] else None,
    }
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    # Save equity curves as CSV
    if results['train']:
        results['train']['df'][['timestamp', 'equity']].to_csv(
            output_path.with_stem(f"{output_path.stem}_train_equity").with_suffix('.csv'),
            index=False
        )
    if results['validation']:
        results['validation']['df'][['timestamp', 'equity']].to_csv(
            output_path.with_stem(f"{output_path.stem}_val_equity").with_suffix('.csv'),
            index=False
        )
    if results['test']:
        results['test']['df'][['timestamp', 'equity']].to_csv(
            output_path.with_stem(f"{output_path.stem}_test_equity").with_suffix('.csv'),
            index=False
        )
    
    print(f"\n  Results saved to: {output_path}")


def main():
    """Main backtesting pipeline."""
    print("=" * 70)
    print("  REAL DATA BITCOIN BACKTESTING PIPELINE")
    print("=" * 70)
    
    # Configuration
    START_DATE = "2021-01-01"
    END_DATE = "2026-03-18"
    TRAIN_RATIO = 0.6
    VALIDATION_RATIO = 0.2
    CAPITAL = 10000.0
    
    print(f"\nConfiguration:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Train/Val/Test split: {TRAIN_RATIO*100:.0f}% / {VALIDATION_RATIO*100:.0f}% / {100-(TRAIN_RATIO+VALIDATION_RATIO)*100:.0f}%")
    print(f"  Initial capital: ${CAPITAL:,.2f}")
    
    # Fetch real Bitcoin hourly data
    print_section("1. FETCHING REAL BITCOIN DATA")
    print("Fetching hourly Bitcoin data from Yahoo Finance...")
    
    try:
        data = fetch_bitcoin_hourly_data(days=365*3)  # ~3 years
        print(f"  Loaded {len(data)} rows")
        print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"  Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    except Exception as e:
        print(f"  ERROR fetching data: {e}")
        print("  Falling back to synthetic data...")
        from quant_algos import generate_minute_bars
        data = generate_minute_bars(n_days=365)
    
    # Split data
    print_section("2. SPLITTING DATA (60/20/20)")
    train_df, val_df, test_df = split_train_validation_test(
        data, 
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )
    print(f"  Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"  Test: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # Strategy 1: Mean Reversion (basic version)
    print_section("3. STRATEGY 1: MEAN REVERSION (BASIC)")
    
    mr_param_grid = {
        'window': [10, 20, 30],
        'std_multiplier': [1.5, 2.0, 2.5],
        'position_size_pct': [0.03, 0.05, 0.10],
        'stop_loss_pct': [0.01, 0.015, 0.02],
        'take_profit_pct': [0.02, 0.025, 0.03],
        'max_holding_period': [30, 60, 120],
    }
    
    mr_opt = optimize_hyperparameters(
        train_df,
        "Mean Reversion",
        mean_reversion_strategy,
        mr_param_grid,
        capital=CAPITAL,
        scoring='combined',
    )
    
    if mr_opt['best_params'] is not None:
        mr_results = run_train_validation_test(
            train_df, val_df, test_df,
            "Mean Reversion",
            mean_reversion_strategy,
            mr_opt['best_params'],
            capital=CAPITAL,
        )
        
        if mr_results:
            save_results(mr_results, "mean_reversion_results")
            
            # Check validation performance
            val_return = mr_results['validation']['metrics']['total_return']
            val_sharpe = mr_results['validation']['metrics']['sharpe_ratio']
            
            if val_return > 0.05 and val_sharpe > 0:  # Positive return and Sharpe > 0
                print("\n  ✓ Mean Reversion strategy passes validation!")
            else:
                print("\n  ✗ Mean Reversion strategy fails validation criteria")
                print(f"    Validation return: {val_return*100:.2f}% (need >5%)")
                print(f"    Validation Sharpe: {val_sharpe:.2f} (need >0)")
    else:
        print("  No valid parameters found for Mean Reversion")
    
    # Strategy 2: Momentum (basic version)
    print_section("4. STRATEGY 2: MOMENTUM (BASIC)")
    
    mom_param_grid = {
        'fast_window': [3, 5, 10],
        'slow_window': [10, 20, 30],
        'volume_threshold': [1.2, 1.5, 2.0],
        'position_size_pct': [0.03, 0.05, 0.10],
        'stop_loss_pct': [0.01, 0.015, 0.02],
        'take_profit_pct': [0.02, 0.03, 0.04],
        'max_holding_period': [30, 60, 120],
    }
    
    mom_opt = optimize_hyperparameters(
        train_df,
        "Momentum",
        momentum_strategy,
        mom_param_grid,
        capital=CAPITAL,
        scoring='combined',
    )
    
    if mom_opt['best_params'] is not None:
        mom_results = run_train_validation_test(
            train_df, val_df, test_df,
            "Momentum",
            momentum_strategy,
            mom_opt['best_params'],
            capital=CAPITAL,
        )
        
        if mom_results:
            save_results(mom_results, "momentum_results")
            
            # Check validation performance
            val_return = mom_results['validation']['metrics']['total_return']
            val_sharpe = mom_results['validation']['metrics']['sharpe_ratio']
            
            if val_return > 0.05 and val_sharpe > 0:
                print("\n  ✓ Momentum strategy passes validation!")
            else:
                print("\n  ✗ Momentum strategy fails validation criteria")
                print(f"    Validation return: {val_return*100:.2f}% (need >5%)")
                print(f"    Validation Sharpe: {val_sharpe:.2f} (need >0)")
    else:
        print("  No valid parameters found for Momentum")
    
    # Summary
    print_section("5. BACKTESTING COMPLETE")
    print("\nResults saved to:", OUTPUT_DIR)
    print("\nNext steps:")
    print("1. Check validation results - did strategy pass?")
    print("2. If pass: Continue to paper trading or production")
    print("3. If fail: Adjust strategy or collect more data")
    
    print("\n" + "=" * 70)
    print("  BACKTESTING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
