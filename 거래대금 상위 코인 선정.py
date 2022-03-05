import time
import pyupbit
import datetime
import pandas as pd

print(datetime.datetime.now())
data = []
coin = []
cryptoc = pyupbit.get_tickers(fiat="KRW")
a = 0
while a < 111:
    df = pyupbit.get_ohlcv(cryptoc[a], interval = "day", count=1)
    v = df.iloc[-1]['value']/1000000
    data.append(v)
    time.sleep(0.05)
    a = a + 1
data.sort()
a = 0
while a < 111:
    df = pyupbit.get_ohlcv(cryptoc[a], interval = "day", count=1)
    if int(data[60]) < int(df.iloc[-1]['value']/1000000):
        coin.append(cryptoc[a])
    time.sleep(0.05)
    a = a + 1
print(coin)
print(len(coin))
print(datetime.datetime.now())
