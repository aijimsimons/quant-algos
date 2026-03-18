"""Mean reversion strategy for short-term trading."""

import pandas as pd
import numpy as np


def mean_reversion_strategy(
    data: pd.DataFrame,
    capital: float = 10000.0,
    window: int = 20,
    std_multiplier: float = 2.0,
    position_size_pct: float = 0.05,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.025,
    min_holding_period: int = 5,
    max_holding_period: int = 60,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Mean reversion strategy using Bollinger Bands.
    
    Entries:
    - Long when price crosses below lower Bollinger Band (oversold)
    - Short when price crosses above upper Bollinger Band (overbought)
    
    Exits:
    - Stop loss or take profit
    - Maximum holding period
    
    Args:
        data: DataFrame with OHLCV data
        capital: Initial capital
        window: Bollinger Bands window
        std_multiplier: Standard deviation multiplier
        position_size_pct: Position size as % of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        min_holding_period: Minimum holding period in bars
        max_holding_period: Maximum holding period in bars
        verbose: Print debug info
        
    Returns:
        DataFrame with strategy signals and P&L
    """
    df = data.copy()
    
    # Reset index to ensure sequential integer indexing
    df = df.reset_index(drop=True)
    
    # Calculate Bollinger Bands
    df['sma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + std_multiplier * df['std']
    df['lower_band'] = df['sma'] - std_multiplier * df['std']
    
    # Calculate Z-score (normalized distance from mean)
    df['zscore'] = (df['close'] - df['sma']) / df['std']
    
    # Generate signals - use tighter thresholds for mean reversion
    df['signal'] = 0
    df.loc[df['zscore'] < -1.0, 'signal'] = 1  # Long
    df.loc[df['zscore'] > 1.0, 'signal'] = -1  # Short
    
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
    
    for i in range(window, n):
        if position == 0:
            # Check for entry - only enter on signal change from 0
            if df.loc[i, 'signal'] == 1:
                # Enter long
                position_value = capital_remaining * position_size_pct
                # Use minimum 1 unit, or calculate based on price
                price = df.loc[i, 'close']
                position = max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < window + 10:
                    print(f"   Entry LONG at i={i}, price={entry_price:.2f}, position={position}")
                
            elif df.loc[i, 'signal'] == -1:
                # Enter short
                position_value = capital_remaining * position_size_pct
                price = df.loc[i, 'close']
                position = -max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < window + 10:
                    print(f"   Entry SHORT at i={i}, price={entry_price:.2f}, position={position}")
                
        else:
            # We have an active position - check for exit
            current_price = df.loc[i, 'close']
            
            # Calculate P&L
            pnl_per_unit = current_price - entry_price if position > 0 else entry_price - current_price
            current_pnl = pnl_per_unit * abs(position)
            
            # Check stop loss
            if current_pnl <= -capital_remaining * stop_loss_pct:
                # Exit on stop loss
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit STOP LOSS at i={i}, pnl={current_pnl:.2f}")
                    
            # Check take profit
            elif current_pnl >= capital_remaining * take_profit_pct:
                # Exit on take profit
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit TAKE PROFIT at i={i}, pnl={current_pnl:.2f}")
                    
            # Check max holding period
            elif i - entry_time >= max_holding_period:
                # Exit on max holding period
                positions[i] = 0
                pnl[i] = current_pnl
                position = 0
                entry_price = 0.0
                if verbose:
                    print(f"   Exit MAX HOLDING at i={i}, pnl={current_pnl:.2f}")
                    
            else:
                # Keep position open
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
    
    # Handle edge cases
    if len(returns) == 0 or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * (252 * 24) ** 0.5
    
    # Calculate equity-based metrics
    equity = df['equity']
    cumulative_pnl = df['cumulative_pnl']
    
    # Max drawdown based on equity
    rolling_max_equity = equity.cummax()
    # Prevent division by zero or negative equity
    safe_rolling_max = rolling_max_equity.replace(0, np.nan).ffill().fillna(1)
    drawdown = np.maximum(0, (rolling_max_equity - equity) / safe_rolling_max)
    max_drawdown = drawdown.max() if len(drawdown) > 1 else 0
    # Cap at 100%
    max_drawdown = min(max_drawdown, 1.0)
    
    # Win rate based on individual trade P&L
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] < 0]['pnl']
    win_rate = len(wins) / len(df[df['pnl'] != 0]) if len(df[df['pnl'] != 0]) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    # Count trades (number of position changes / 2)
    position_changes = df['position'].diff().abs().sum()
    total_trades = int(position_changes // 2) if position_changes > 0 else 0
    
    # Average win/loss per trade
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


def enhanced_mean_reversion_strategy(
    data: pd.DataFrame,
    capital: float = 10000.0,
    window: int = 20,
    std_multiplier: float = 2.0,
    position_size_pct: float = 0.05,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.025,
    min_holding_period: int = 5,
    max_holding_period: int = 60,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    use_rsi_filter: bool = True,
    use_trend_filter: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Enhanced mean reversion strategy with RSI and trend filters.
    
    This strategy combines:
    - Bollinger Bands for mean reversion signals
    - RSI to avoid entering during strong trends
    - Trend filter to only take mean reversion in range-bound markets
    
    Args:
        data: DataFrame with OHLCV data (should have technical indicators)
        capital: Initial capital
        window: Bollinger Bands window
        std_multiplier: Standard deviation multiplier
        position_size_pct: Position size as % of capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        min_holding_period: Minimum holding period in bars
        max_holding_period: Maximum holding period in bars
        rsi_oversold: RSI threshold for oversold conditions
        rsi_overbought: RSI threshold for overbought conditions
        use_rsi_filter: Whether to use RSI filter for entries
        use_trend_filter: Whether to use trend filter (only mean revert when trend is weak)
        verbose: Print debug info
        
    Returns:
        DataFrame with strategy signals and P&L
    """
    df = data.copy()
    
    # Ensure we have the required columns
    required_cols = ['sma', 'std', 'upper_band', 'lower_band', 'zscore', 'rsi']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Run add_technical_indicators() first.")
    
    # Generate signals with enhanced filters
    df['signal'] = 0
    
    # Base mean reversion signals
    long_base = df['zscore'] < -1.0  # Price is below lower band
    short_base = df['zscore'] > 1.0  # Price is above upper band
    
    # Add RSI filter if enabled
    if use_rsi_filter:
        # For long entries, RSI should not be too high (avoid buying overbought)
        long_rsi = long_base & (df['rsi'] < rsi_overbought)
        # For short entries, RSI should not be too low (avoid selling oversold)
        short_rsi = short_base & (df['rsi'] > rsi_oversold)
    else:
        long_rsi = long_base
        short_rsi = short_base
    
    # Add trend filter if enabled
    if use_trend_filter:
        # Only mean revert when trend is weak (price near moving average)
        trend_strength = abs(df['close'] - df['sma']) / df['sma']
        trend_filter = trend_strength < 0.02  # Price within 2% of SMA
        long_trend = long_rsi & trend_filter
        short_trend = short_rsi & trend_filter
    else:
        long_trend = long_rsi
        short_trend = short_rsi
    
    # Apply signals
    df.loc[long_trend, 'signal'] = 1
    df.loc[short_trend, 'signal'] = -1
    
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
    
    for i in range(window, n):
        if position == 0:
            # Check for entry - only enter on signal change from 0
            if df.loc[i, 'signal'] == 1:
                # Enter long
                position_value = capital_remaining * position_size_pct
                price = df.loc[i, 'close']
                position = max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < window + 10:
                    print(f"   Entry LONG at i={i}, price={entry_price:.2f}, position={position}")
                
            elif df.loc[i, 'signal'] == -1:
                # Enter short
                position_value = capital_remaining * position_size_pct
                price = df.loc[i, 'close']
                position = -max(1, int(position_value / price))
                entry_price = price
                entry_time = i
                if verbose and i < window + 10:
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
