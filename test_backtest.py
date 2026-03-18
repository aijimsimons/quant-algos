"""Test script for backtesting pipeline."""

from quant_algos import generate_sample_data, momentum_strategy, BacktestEngine


def test_pipeline():
    """Test the full backtesting pipeline."""
    print("Generating sample data...")
    data = generate_sample_data(n_days=100, start_price=70000.0)
    print(f"Data shape: {data.shape}")
    print(data.head())
    
    print("\nRunning momentum strategy...")
    results = momentum_strategy(data, capital=10000.0)
    print(f"Results shape: {results.shape}")
    print(results[['timestamp', 'price', 'signal', 'cumulative_returns']].tail())
    
    print("\nRunning backtest engine...")
    engine = BacktestEngine(data, capital=10000.0)
    results = engine.run(momentum_strategy)
    metrics = engine.get_metrics()
    
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    test_pipeline()
