"""Test momentum strategy."""

from quant_algos import generate_minute_bars, momentum_strategy, calculate_metrics


def test_momentum():
    """Test momentum strategy with minute data."""
    print("=" * 60)
    print("Testing Momentum Strategy (Short-Term)")
    print("=" * 60)
    
    data = generate_minute_bars(
        n_days=30,
        start_price=70000.0,
        volatility=0.005,
        drift=0.00005,
    )
    
    print(f"\nData shape: {data.shape}")
    
    # Run strategy
    results = momentum_strategy(
        data,
        capital=10000.0,
        fast_window=5,
        slow_window=20,
        volume_threshold=1.5,
        position_size_pct=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03,
        max_holding_period=60,
        verbose=False,
    )
    
    # Debug signals
    print("\nSignal distribution:")
    print(f"  Long signals: {(results['signal'] == 1).sum()}")
    print(f"  Short signals: {(results['signal'] == -1).sum()}")
    print(f"  Volume MA range: {results['volume_ma'].min():.0f} - {results['volume_ma'].max():.0f}")
    print(f"  Volume range: {results['volume'].min():.0f} - {results['volume'].max():.0f}")
    
    metrics = calculate_metrics(results, capital=10000.0)
    
    print("\nPerformance Metrics:")
    print(f"   Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics['total_trades']}")
    
    print(f"\nFinal Equity: ${metrics['total_return']*10000 + 10000:,.2f}")
    
    return results, metrics


if __name__ == "__main__":
    test_momentum()
