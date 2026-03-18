#!/usr/bin/env python3
"""Test train/validation/test split with real data pipeline."""

from quant_algos import (
    generate_minute_bars,
    add_technical_indicators,
    split_train_validation_test,
    mean_reversion_strategy,
    calculate_metrics,
)

def test_train_val_test_split():
    print("=" * 60)
    print("TRAIN/VALIDATION/TEST SPLIT TEST")
    print("=" * 60)
    
    # Generate 30 days of data
    print("\nGenerating 30 days of synthetic data...")
    data = generate_minute_bars(n_days=30)
    print(f"Total data: {len(data)} rows")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Split data
    print("\nSplitting data (60/20/20)...")
    train_df, val_df, test_df = split_train_validation_test(data, train_ratio=0.6, validation_ratio=0.2)
    print(f"  Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"  Test: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    # Train hyperparameter optimization (simplified - just test a few params)
    print("\nTesting Mean Reversion on TRAIN data...")
    train_with_indicators = add_technical_indicators(train_df)
    
    # Test a few parameter combinations
    test_params = [
        {'window': 20, 'std_multiplier': 2.0, 'max_holding_period': 60},
        {'window': 30, 'std_multiplier': 2.5, 'max_holding_period': 120},
    ]
    
    best_score = -float('inf')
    best_params = None
    
    for params in test_params:
        result = mean_reversion_strategy(train_with_indicators.copy(), capital=10000.0, **params)
        metrics = calculate_metrics(result, capital=10000.0)
        
        # Combined score: Sharpe with penalty for drawdown
        score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
        
        print(f"\n  Params: {params}")
        print(f"    Return: {metrics['total_return']*100:.2f}%")
        print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"    Max DD: {metrics['max_drawdown']*100:.2f}%")
        print(f"    Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    # Validation (single run, no further tuning!)
    print("\n" + "-" * 60)
    print("VALIDATION (NO FURTHER TUNING ALLOWED)")
    print("-" * 60)
    
    val_with_indicators = add_technical_indicators(val_df)
    val_result = mean_reversion_strategy(
        val_with_indicators,
        capital=10000.0,
        **best_params
    )
    val_metrics = calculate_metrics(val_result, capital=10000.0)
    
    print(f"\nValidation Metrics:")
    print(f"  Return: {val_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {val_metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {val_metrics['max_drawdown']*100:.2f}%")
    print(f"  Trades: {val_metrics['total_trades']}")
    
    # Validation check
    if val_metrics['total_return'] > 0.05 and val_metrics['sharpe_ratio'] > 0:
        print("\n  ✓ VALIDATION PASSED - Strategy shows promise!")
        print("  Proceeding to test set evaluation...")
    else:
        print("\n  ✗ VALIDATION FAILED")
        print("  Strategy does not meet criteria (need >5% return and Sharpe > 0)")
        print("  Discarding strategy or collecting more data.")
    
    # Test (final evaluation)
    print("\n" + "-" * 60)
    print("TEST SET EVALUATION")
    print("-" * 60)
    
    test_with_indicators = add_technical_indicators(test_df)
    test_result = mean_reversion_strategy(
        test_with_indicators,
        capital=10000.0,
        **best_params
    )
    test_metrics = calculate_metrics(test_result, capital=10000.0)
    
    print(f"\nTest Metrics:")
    print(f"  Return: {test_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {test_metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {test_metrics['max_drawdown']*100:.2f}%")
    print(f"  Trades: {test_metrics['total_trades']}")
    
    print("\n" + "=" * 60)
    print("TRAIN/VALIDATION/TEST PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_train_val_test_split()
