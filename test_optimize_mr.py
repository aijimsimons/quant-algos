"""Optimize mean reversion strategy parameters."""

from quant_algos import generate_minute_bars, mean_reversion_strategy, calculate_metrics


def test_parameter_grid():
    """Test different parameter combinations."""
    print("=" * 70)
    print("Mean Reversion Strategy Parameter Optimization")
    print("=" * 70)
    
    # Generate data
    data = generate_minute_bars(
        n_days=30,
        start_price=70000.0,
        volatility=0.005,
        drift=0.00005,
    )
    
    # Parameter grid
    windows = [10, 20, 30]
    std_mults = [1.5, 2.0, 2.5]
    position_sizes = [0.03, 0.05, 0.10]
    stop_losses = [0.01, 0.015, 0.02]
    take_profits = [0.02, 0.025, 0.03]
    
    best_score = -float('inf')
    best_params = None
    results = []
    
    for window in windows:
        for std_mult in std_mults:
            for pos_size in position_sizes:
                for stop_loss in stop_losses:
                    for take_profit in take_profits:
                        try:
                            result = mean_reversion_strategy(
                                data,
                                capital=10000.0,
                                window=window,
                                std_multiplier=std_mult,
                                position_size_pct=pos_size,
                                stop_loss_pct=stop_loss,
                                take_profit_pct=take_profit,
                                max_holding_period=60,
                            )
                            
                            metrics = calculate_metrics(result, capital=10000.0)
                            
                            # Score: Sharpe ratio with penalty for max drawdown
                            score = metrics['sharpe_ratio'] - 0.5 * metrics['max_drawdown']
                            
                            results.append({
                                'window': window,
                                'std_mult': std_mult,
                                'pos_size': pos_size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'total_return': metrics['total_return'],
                                'sharpe_ratio': metrics['sharpe_ratio'],
                                'max_drawdown': metrics['max_drawdown'],
                                'win_rate': metrics['win_rate'],
                                'profit_factor': metrics['profit_factor'],
                                'total_trades': metrics['total_trades'],
                                'score': score,
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'window': window,
                                    'std_mult': std_mult,
                                    'pos_size': pos_size,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                }
                        except Exception as e:
                            continue
    
    print(f"\nBest parameters found:")
    print(f"  Window: {best_params['window']}")
    print(f"  Std Multiplier: {best_params['std_mult']}")
    print(f"  Position Size: {best_params['pos_size']*100}%")
    print(f"  Stop Loss: {best_params['stop_loss']*100}%")
    print(f"  Take Profit: {best_params['take_profit']*100}%")
    print(f"\nBest score: {best_score:.4f}")
    
    # Show top 5 results
    print("\nTop 5 parameter combinations:")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False).head(5)
    
    for _, row in results_df.iterrows():
        print(f"\n  Window={row['window']}, StdMult={row['std_mult']}, "
              f"PosSize={row['pos_size']*100:.0f}%")
        print(f"  Stop={row['stop_loss']*100:.0f}%, Take={row['take_profit']*100:.0f}%")
        print(f"  Return={row['total_return']*100:.2f}%, Sharpe={row['sharpe_ratio']:.2f}, "
              f"MaxDD={row['max_drawdown']*100:.2f}%, WinRate={row['win_rate']*100:.1f}%")
    
    return best_params, best_score


if __name__ == "__main__":
    import pandas as pd
    test_parameter_grid()
