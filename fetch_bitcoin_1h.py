
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'BTC/USDT'
timeframe = '1h'
days = 180

end_time = datetime.now()
start_time = end_time - timedelta(days=days)

print(f"Fetching {timeframe} Bitcoin data for last {days} days...")

all_ohlcvs = []
current_since = int(start_time.timestamp() * 1000)
end_since = int(end_time.timestamp() * 1000)
max_iterations = 200  # Limit iterations

iteration = 0
while current_since < end_since and iteration < max_iterations:
    for attempt in range(3):
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
            if ohlcvs:
                all_ohlcvs.extend(ohlcvs)
                last_timestamp = ohlcvs[-1][0]
                current_since = last_timestamp + 60000
                break
            else:
                break
        except Exception as e:
            if attempt == 2:
                break
            time.sleep(1)
    
    iteration += 1
    if iteration % 50 == 0:
        print(f"  Iteration {iteration}, fetched {len(all_ohlcvs)} candles so far...")

if not all_ohlcvs:
    print("No data fetched!")
else:
    df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    # Save to CSV
    df.to_csv('/Users/xingjianliu/jim/quant-algos/data/btc_1h_180d.csv', index=False)
    print("Data saved to data/btc_1h_180d.csv")
