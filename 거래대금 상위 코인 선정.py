import pyupbit
import time
import pandas as pd

def get_coin():
    data = []
    coin = []
    cryptoc = pyupbit.get_tickers(fiat="KRW")
    a = 0
    while a < 110:
        df = pyupbit.get_ohlcv(cryptoc[a], interval = "day", count = 12)
        ma = df['close'].rolling(window=10).mean()
        v = df.iloc[-1]['volume'] * ma.iloc[-1]
        data.append(v)
        a = a + 1
        time.sleep(0.1)
    data.sort()
    middle = int(data[50])

    cryptoc = pyupbit.get_tickers(fiat="KRW")
    a = 0
    while a < 110:
        df = pyupbit.get_ohlcv(cryptoc[a], interval = "day", count = 12)
        ma = df['close'].rolling(window=10).mean()
        k = int(df.iloc[-1]['volume'])
        p = int(ma.iloc[-1])
        if k * p > middle:
            coin.append(cryptoc[a])
        a = a + 1
        time.sleep(0.1)
    return coin
coin = get_coin()
