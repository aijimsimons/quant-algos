"""Backtesting engine for trading strategies."""

import pandas as pd
from typing import Callable


class BacktestEngine:
    """Simple backtesting engine for strategy evaluation."""
    
    def __init__(self, data: pd.DataFrame, capital: float = 10000.0):
        """
        Initialize backtest engine.
        
        Args:
            data: DataFrame with price data
            capital: Initial capital in USD
        """
        self.data = data.copy()
        self.capital = capital
        self.results = None
        
    def run(self, strategy_func: Callable, **kwargs) -> pd.DataFrame:
        """
        Run backtest on given strategy.
        
        Args:
            strategy_func: Strategy function that returns DataFrame with signals
            **kwargs: Arguments to pass to strategy function
            
        Returns:
            DataFrame with strategy results
        """
        self.results = strategy_func(self.data, capital=self.capital, **kwargs)
        return self.results
    
    def get_metrics(self) -> dict:
        """Calculate performance metrics from backtest results."""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        returns = self.results['strategy_returns'].dropna()
        
        metrics = {
            'total_return': self.results['cumulative_returns'].iloc[-1] - 1,
            'sharpe_ratio': (returns.mean() / returns.std()) * (252 ** 0.5),
            'max_drawdown': (self.results['cumulative_returns'].cummax() - self.results['cumulative_returns']).max(),
            'win_rate': (returns > 0).mean(),
            'trades': (self.results['signal'].diff() != 0).sum(),
        }
        
        return metrics
