#!/usr/bin/env python3
import yfinance

ticker = yfinance.Ticker('BTC-USD')
data = ticker.history(period='1mo', interval='1h')

print('Columns:', list(data.columns))
print('Index:', data.index.name)
print(data.head())
print(data.tail())
