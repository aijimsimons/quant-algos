#!/usr/bin/env python3
"""Quick test to verify the strategy pipeline works."""

from quant_algos import (
    generate_minute_bars,
    add_technical_indicators,
    mean_reversion_strategy,
    momentum_strategy,
    calculate_metrics,
)

def test_mean_reversion():
    print("Testing Mean Reversion Strategy...")
    
    # Generate small dataset
    data = generate_minute_bars(n_days=1)
    print(f"Generated data shape: {data.shape}")
    
    # Add technical indicators
    data = add_technical_indicators(data)
    print(f"After adding indicators: {data.shape}")
    
    # Run strategy
    result = mean_reversion_strategy(
        data,
        capital=10000.0,
        window=20,
        std_multiplier=2.0,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.025,
        max_holding_period=30,
    )
    print(f"Result shape: {result.shape}")
    
    # Calculate metrics
    metrics = calculate_metrics(result, capital=10000.0)
    print(f"\nMean Reversion Metrics:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    
    return metrics

def test_momentum():
    print("\nTesting Momentum Strategy...")
    
    # Generate small dataset
    data = generate_minute_bars(n_days=1)
    print(f"Generated data shape: {data.shape}")
    
    # Add technical indicators
    data = add_technical_indicators(data)
    print(f"After adding indicators: {data.shape}")
    
    # Run strategy
    result = momentum_strategy(
        data,
        capital=10000.0,
        fast_window=5,
        slow_window=20,
        volume_threshold=1.5,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        max_holding_period=30,
    )
    print(f"Result shape: {result.shape}")
    
    # Calculate metrics
    metrics = calculate_metrics(result, capital=10000.0)
    print(f"\nMomentum Metrics:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    
    return metrics

if __name__ == "__main__":
    mr_metrics = test_mean_reversion()
    mom_metrics = test_momentum()
    
    print("\n" + "=" * 50)
    print("QUICK TEST COMPLETE")
    print("=" * 50)
