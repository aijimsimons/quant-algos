#!/usr/bin/env python3
"""Training with risk management - limit drawdown exposure."""

from quant_algos import (
    generate_minute_bars,
    add_technical_indicators,
    mean_reversion_strategy,
    calculate_metrics,
)

import pandas as pd
import numpy as np
from pathlib import Path
import json

OUTPUT_DIR = Path("train_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def mean_reversion_strategy_risk_managed(
    data: pd.DataFrame,
    capital: float = 10000.0,
    window: int = 20,
    std_multiplier: float = 2.0,
    position_size_pct: float = 0.05,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.025,
    max_holding_period: int = 60,
    max_drawdown_pct: float = 0.10,  # Max 10% drawdown
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Mean reversion strategy with dynamic position sizing based on drawdown.
    
    Reduces position size as drawdown increases to manage risk.
    """
    df = data.copy()
    df = df.reset_index(drop=True)
    
    # Calculate Bollinger Bands
    df['sma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + std_multiplier * df['std']
    df['lower_band'] = df['sma'] - std_multiplier * df['std']
    df['zscore'] = (df['close'] - df['sma']) / df['std']
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['zscore'] < -1.0, 'signal'] = 1
    df.loc[df['zscore'] > 1.0, 'signal'] = -1
    
    # Pre-allocate arrays
    n = len(df)
    positions = np.zeros(n, dtype=np.float64)
    entry_prices = np.zeros(n, dtype=np.float64)
    entry_times = np.zeros(n, dtype=np.float64)
    pnl = np.zeros(n, dtype=np.float64)
    
    # Strategy state
    capital_remaining = capital
    position = 0
    entry_price = 0.0
    entry_time = 0
    peak_equity = capital  # Track peak for drawdown calculation
    
    for i in range(window, n):
        if position == 0:
            if df.loc[i, 'signal'] == 1:
                # Dynamic position sizing based on drawdown
                drawdown = (peak_equity - capital_remaining) / peak_equity
                # Reduce position as drawdown increases
                if drawdown > 0:
                    dd_factor = 1.0 - (drawdown / max_drawdown_pct)
                    dd_factor = max(0.1, dd_factor)  # Min 10% position
                else:
                    dd_factor = 1.0
                
                position_value = capital_remaining * position_size_pct * dd_factor
                price = df.loc[i, 'close']
                position = max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                
            elif df.loc[i, 'signal'] == -1:
                drawdown = (peak_equity - capital_remaining) / peak_equity
                if drawdown > 0:
                    dd_factor = 1.0 - (drawdown / max_drawdown_pct)
                    dd_factor = max(0.1, dd_factor)
                else:
                    dd_factor = 1.0
                
                position_value = capital_remaining * position_size_pct * dd_factor
                price = df.loc[i, 'close']
                position = -max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                
        else:
            current_price = df.loc[i, 'close']
            pnl_per_unit = current_price - entry_price if position > 0 else entry_price - current_price
            current_pnl = pnl_per_unit * abs(position)
            
            # Update peak equity
            current_equity = capital_remaining + current_pnl
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # Check drawdown limit
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown >= max_drawdown_pct:
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit DRAWDOWN LIMIT at i={i}, pnl={current_pnl:.2f}, dd={drawdown*100:.2f}%")
                continue
            
            # Check stop loss
            if current_pnl <= -capital_remaining * stop_loss_pct:
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit STOP LOSS at i={i}, pnl={current_pnl:.2f}")
                    
            # Check take profit
            elif current_pnl >= capital_remaining * take_profit_pct:
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit TAKE PROFIT at i={i}, pnl={current_pnl:.2f}")
                    
            # Check max holding period
            elif i - entry_time >= max_holding_period:
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit MAX HOLDING at i={i}, pnl={current_pnl:.2f}")
                    
            else:
                positions[i] = float(position)
                entry_prices[i] = entry_price
                entry_times[i] = float(entry_time)
    
    # Assign to DataFrame
    df['position'] = positions
    df['entry_price'] = entry_prices
    df['entry_time'] = entry_times
    df['pnl'] = pnl
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = capital + df['cumulative_pnl']
    df['returns'] = df['equity'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    
    return df


def run_training_with_risk_management():
    """Run training with risk-managed strategy."""
    print("=" * 60)
    print("  RISK-MANAGED MEAN REVERSION")
    print("=" * 60)
    
    data = generate_minute_bars(n_days=7)
    print(f"\nData: {len(data)} rows")
    
    data = add_technical_indicators(data)
    
    # Parameters
    params = {
        'window': 25,
        'std_multiplier': 2.0,
        'position_size_pct': 0.05,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.025,
        'max_holding_period': 60,
        'max_drawdown_pct': 0.10,  # 10% max drawdown limit
    }
    
    result = mean_reversion_strategy_risk_managed(
        data.copy(), capital=10000.0, **params
    )
    
    metrics = calculate_metrics(result, capital=10000.0)
    
    print("\nPerformance Metrics (with 10% drawdown limit):")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    
    return metrics


if __name__ == "__main__":
    metrics = run_training_with_risk_management()
