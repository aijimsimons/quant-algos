#!/usr/bin/env python3
"""Test real data fetching."""

from quant_algos import fetch_bitcoin_hourly_data, add_technical_indicators, mean_reversion_strategy, calculate_metrics

print("Testing real Bitcoin data fetching...")

try:
    data = fetch_bitcoin_hourly_data(days=30)
    print(f"Loaded {len(data)} rows")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    data = add_technical_indicators(data)
    print(f"After indicators: {len(data.columns)} columns")
    
    result = mean_reversion_strategy(data.copy(), capital=10000.0, window=20, std_multiplier=2.0, position_size_pct=0.05, stop_loss_pct=0.02, take_profit_pct=0.025, max_holding_period=60)
    metrics = calculate_metrics(result, capital=10000.0)
    
    print(f"\nPerformance:")
    print(f"  Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Trades: {metrics['total_trades']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
