"""Momentum breakout strategy for short-term trading."""

import pandas as pd
import numpy as np


def momentum_strategy(
    data: pd.DataFrame,
    capital: float = 10000.0,
    fast_window: int = 5,
    slow_window: int = 20,
    volume_threshold: float = 1.5,
    position_size_pct: float = 0.05,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.03,
    max_holding_period: int = 60,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Momentum breakout strategy using moving average crossover and volume confirmation.
    
    Entries:
    - Long when price breaks above slow MA with high volume
    - Short when price breaks below slow MA with high volume
    
    Args:
        data: DataFrame with OHLCV data
        capital: Initial capital
        fast_window: Fast MA window
        slow_window: Slow MA window  
        volume_threshold: Volume multiplier above average to confirm breakout
        position_size_pct: Position size as % of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        max_holding_period: Maximum holding period in bars
        verbose: Print debug info
        
    Returns:
        DataFrame with strategy signals and P&L
    """
    df = data.copy()
    
    # Calculate moving averages
    df['fast_ma'] = df['close'].rolling(window=fast_window).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_window).mean()
    
    # Calculate volume average
    df['volume_ma'] = df['volume'].rolling(window=slow_window).mean()
    
    # Calculate price momentum
    df['momentum'] = df['close'].pct_change(periods=fast_window)
    
    # Calculate volatility
    df['volatility'] = df['close'].rolling(window=slow_window).std()
    
    # Generate signals
    df['signal'] = 0
    
    # Long signal: price above slow MA, fast MA above slow MA, high volume
    df.loc[
        (df['close'] > df['slow_ma']) &
        (df['fast_ma'] > df['slow_ma']) &
        (df['volume'] > volume_threshold * df['volume_ma']) &
        (df['momentum'] > 0),
        'signal'
    ] = 1
    
    # Short signal: price below slow MA, fast MA below slow MA, high volume
    df.loc[
        (df['close'] < df['slow_ma']) &
        (df['fast_ma'] < df['slow_ma']) &
        (df['volume'] > volume_threshold * df['volume_ma']) &
        (df['momentum'] < 0),
        'signal'
    ] = -1
    
    # Pre-allocate arrays for performance
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
    
    for i in range(slow_window, n):
        if position == 0:
            # Check for entry
            if df.loc[i, 'signal'] == 1:
                # Enter long
                position_value = capital_remaining * position_size_pct
                price = df.loc[i, 'close']
                position = max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < slow_window + 20:
                    print(f"   Entry LONG at i={i}, price={entry_price:.2f}, position={position}")
                
            elif df.loc[i, 'signal'] == -1:
                # Enter short
                position_value = capital_remaining * position_size_pct
                price = df.loc[i, 'close']
                position = -max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < slow_window + 20:
                    print(f"   Entry SHORT at i={i}, price={entry_price:.2f}, position={position}")
                
        else:
            # We have an active position - check for exit
            current_price = df.loc[i, 'close']
            
            # Calculate P&L
            pnl_per_unit = current_price - entry_price if position > 0 else entry_price - current_price
            current_pnl = pnl_per_unit * abs(position)
            
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
    
    # Calculate cumulative P&L
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = capital + df['cumulative_pnl']
    df['returns'] = df['equity'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    
    return df


def calculate_metrics(df: pd.DataFrame, capital: float = 10000.0) -> dict:
    """Calculate strategy performance metrics."""
    returns = df['returns'].dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * (252 * 24) ** 0.5
    
    equity = df['equity']
    rolling_max_equity = equity.cummax()
    drawdown = (rolling_max_equity - equity) / rolling_max_equity
    max_drawdown = drawdown.max() if len(drawdown) > 1 else 0
    
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] < 0]['pnl']
    win_rate = len(wins) / len(df[df['pnl'] != 0]) if len(df[df['pnl'] != 0]) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    position_changes = df['position'].diff().abs().sum()
    total_trades = int(position_changes // 2) if position_changes > 0 else 0
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    
    metrics = {
        'total_return': (equity.iloc[-1] - capital) / capital if len(equity) > 1 else 0,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
    }
    
    return metrics
