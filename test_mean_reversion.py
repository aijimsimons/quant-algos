"""Test mean reversion strategy."""

from quant_algos import generate_minute_bars, mean_reversion_strategy, calculate_metrics
from quant_algos.backtest import BacktestEngine


def test_mean_reversion():
    """Test mean reversion strategy with minute data."""
    print("=" * 60)
    print("Testing Mean Reversion Strategy (Short-Term)")
    print("=" * 60)
    
    # Generate 30 days of minute data
    print("\n1. Generating 30 days of minute-by-minute data...")
    data = generate_minute_bars(
        n_days=30,
        start_price=70000.0,
        volatility=0.005,
        drift=0.00005,
    )
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"   Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    # Run strategy with short timeframes
    print("\n2. Running mean reversion strategy (20-period Bollinger Bands)...")
    results = mean_reversion_strategy(
        data,
        capital=10000.0,
        window=20,  # 20 minutes
        std_multiplier=2.0,
        position_size_pct=0.05,  # 5% of capital per trade
        stop_loss_pct=0.015,  # 1.5% stop loss
        take_profit_pct=0.025,  # 2.5% take profit
        max_holding_period=60,  # Max 60 minutes
        verbose=True,  # Debug output
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results, capital=10000.0)
    
    print("\n3. Performance Metrics:")
    print(f"   Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Avg Win: ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss: ${metrics['avg_loss']:.2f}")
    
    # Final equity
    print(f"\n4. Final Equity: ${metrics['total_return']*10000 + 10000:,.2f}")
    
    # Sample trades
    trades = results[results['position'].diff() != 0]
    if len(trades) > 0:
        print("\n5. Sample Trades (first 5):")
        for _, trade in trades.head().iterrows():
            pos = int(trade['position'])
            side = "LONG" if pos > 0 else "SHORT" if pos < 0 else "EXIT"
            print(f"   {trade['timestamp']}: {side} @ ${trade['entry_price']:.0f} | PnL: ${trade['pnl']:.2f}")
    else:
        print("\n5. No trades executed - strategy signals not triggering")
    
    return results, metrics


if __name__ == "__main__":
    test_mean_reversion()
