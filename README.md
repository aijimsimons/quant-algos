# quant-algos

Collection of trading strategies for quantitative crypto trading.

## Strategy Categories

- **Statistical Arbitrage**: Pair trading, mean reversion, cointegration
- **Machine Learning**: LSTM, Transformer, GNN on order book
- **Regime Detection**: HMM, change-point detection
- **Execution**: TWAP, VWAP, impact models
- **Fundamental**: On-chain metrics, exchange flows
- **Sentiment**: News, social media, funding rates

## Usage

```python
from quant_algos import strategies

# Load strategy
strategy = strategies.momentum_strategy()

# Backtest
result = engine.backtest(data, strategy)
```
