import plotly.graph_objects as go
import plotly.subplots as ms
import plotly.express as px
import pyupbit
import pandas as pd
import numpy as np
import datetime

print(datetime.datetime.now())
ticker = "krw-xrp"
interday = "minute60"
count_lenth = 1000
next = 100

df = pyupbit.get_ohlcv(ticker, interval=interday, count=count_lenth)
data = pyupbit.get_ohlcv(ticker, interval=interday, count=next)
close = data['close']

base = (close - np.min(close)) / (np.max(close) - np.min(close))
w = len(base)
move = len(df) - w - next - 1

sim = []
for i in range(move):
  t = df.iloc[i:i+w]['close']
  base2 = (t - np.min(t)) / (np.max(t) - np.min(t))
  a = np.dot(base, base2)/(np.sqrt(np.dot(base, base)) * np.sqrt(np.dot(base2, base2)))
  sim.append(a)

ser = pd.Series(sim).sort_values(ascending=False).head(1)

predict = round(ser.iloc[0],4)
ser = ser.index

i = ser[0]
chart = df.iloc[i:i+w+next]
gap = chart.iloc[next]['open'] - data.iloc[-1]['close']
chart = chart - gap
chart['volume'] = chart['volume'] + gap
def get_chart(c):
  a = 0
  while a < next:
    chart.iloc[a][c] = data.iloc[a][c]
    a = a + 1
  return chart
get_chart("close")
get_chart("open")
get_chart("high")
get_chart("low")

df = chart
df['ma20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['upper'] = df['ma20'] + 2*df['stddev']
df['lower'] = df['ma20'] - 2*df['stddev']

df['ma12'] = df['close'].rolling(window=12).mean()
df['ma26'] = df['close'].rolling(window=26).mean()
df['MACD'] = df['ma12'] - df['ma26']
df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
df['MACD_Oscil'] = df['MACD'] - df['MACD_Signal']

df['ndays_high'] = df['high'].rolling(window=14, min_periods=1).max()
df['ndays_low'] = df['low'].rolling(window=14, min_periods=1).min()
df['fast_k'] = (df['close'] - df['ndays_low']) / (df['ndays_high'] - df['ndays_low']) * 100
df['slow_d'] = df['fast_k'].rolling(window=3).mean()

df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
df['TP'] = (df['high'] + df['low'] + df['close']) / 3
df['PMF'] = 0
df['NMF'] = 0
for i in range(len(df.close)-1):
  if df.TP.values[i] < df.TP.values[i+1]:
    df.PMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
    df.NMF.values[i+1] = 0
  else:
    df.NMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
    df.PMF.values[i+1] = 0
  df['MFR'] = (df.PMF.rolling(window=10).sum() / df.NMF.rolling(window=10).sum())
  df['MFI10'] = 100 - 100 / (1 + df['MFR'])

U = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
D = np.where(df['close'].diff(1) < 0, df['close'].diff(1) *(-1), 0)
AU = pd.DataFrame(U, index=df.index).rolling(window=14).mean()
AD = pd.DataFrame(D, index=df.index).rolling(window=14).mean()
RSI = AU / (AD+AU) *100
df['RSI'] = RSI

df = df[25:]
print(datetime.datetime.now())

candle = go.Candlestick(open=df['open'],high=df['high'],low=df['low'],close=df['close'], increasing_line_color = 'green',decreasing_line_color = 'red', showlegend=False)
upper = go.Scatter(y=df['upper'], line=dict(color='red', width=2), name='upper', showlegend=False)
ma20 = go.Scatter(y=df['ma20'], line=dict(color='black', width=2), name='ma20', showlegend=False)
lower = go.Scatter(y=df['lower'], line=dict(color='blue', width=2), name='lower', showlegend=False)
Volume = go.Bar(y=df['volume'], marker_color='red', name='Volume', showlegend=False)
MACD = go.Scatter(y=df['MACD'], line=dict(color='blue', width=2), name='MACD', legendgroup='group2', legendgrouptitle_text='MACD')
MACD_Signal = go.Scatter(y=df['MACD_Signal'], line=dict(dash='dashdot', color='green', width=2), name='MACD_Signal')
MACD_Oscil = go.Bar(y=df['MACD_Oscil'], marker_color='purple', name='MACD_Oscil')
fast_k = go.Scatter(y=df['fast_k'], line=dict(color='skyblue', width=2), name='fast_k', legendgroup='group3', legendgrouptitle_text='%K %D')
slow_d = go.Scatter(y=df['slow_d'], line=dict(dash='dashdot', color='black', width=2), name='slow_d')
PB = go.Scatter(y=df['PB']*100, line=dict(color='blue', width=2), name='PB', legendgroup='group4', legendgrouptitle_text='PB, MFI')
MFI10 = go.Scatter(y=df['MFI10'], line=dict(dash='dashdot', color='green', width=2), name='MFI10')
RSI = go.Scatter(y=df['RSI'], line=dict(color='red', width=2), name='RSI', legendgroup='group5', legendgrouptitle_text='RSI')

fig = ms.make_subplots(rows=5, cols=2, specs=[[{'rowspan':4},{}],[None,{}],[None,{}],[None,{}],[{},{}]], shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.01)
fig.add_trace(candle,row=1,col=1)
fig.add_trace(upper,row=1,col=1)
fig.add_trace(ma20,row=1,col=1)
fig.add_trace(lower,row=1,col=1)
fig.add_trace(Volume,row=5,col=1)
fig.add_trace(candle,row=1,col=2)
fig.add_trace(upper,row=1,col=2)
fig.add_trace(ma20,row=1,col=2) 
fig.add_trace(lower,row=1,col=2)
fig.add_trace(MACD,row=2,col=2)
fig.add_trace(MACD_Signal,row=2,col=2)
fig.add_trace(MACD_Oscil,row=2,col=2)
fig.add_trace(fast_k,row=3,col=2)
fig.add_trace(slow_d,row=3,col=2)
fig.add_trace(PB,row=4,col=2)
fig.add_trace(MFI10,row=4,col=2)
fig.add_trace(RSI,row=5,col=2)

fig.update_layout(autosize=True, xaxis1_rangeslider_visible=False, xaxis2_rangeslider_visible=False, margin=dict(l=50,r=50,t=50,b=50), template='seaborn', title='비트코인 선물 가격 예측')
fig.update_xaxes(tickformat='%y년%m월%d일', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2, gridcolor='lightgray', showline=True,linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(tickformat=',d', zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2, gridcolor='lightgray',showline=True,linewidth=2, linecolor='black', mirror=True)
fig.update_traces(xhoverformat='%y년%m월%d일')
fig.show()
