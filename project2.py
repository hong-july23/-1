#모듈 임포팅
from logging import exception
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import FinanceDataReader as fdr
from bs4 import BeautifulSoup
import OpenDartReader
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ms
import ccxt
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import webbrowser
import requests
import test

#객체 생성
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
    #UI 메인화면 디자인
    def setupUI(self):
        #위젯 생성
        self.move(0,0)
        self.resize(1920, 1440)
        #self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Traider assistant(stocks, indices, crypto...)")
        self.setWindowIcon(QIcon('btc.png'))

        self.stockname = QLabel(self)
        self.stocksector = QLabel(self)
        self.stockindustry = QLabel(self)
        self.stockprice = QLabel(self)
        self.stockchigh = QLabel(self) 
        self.stocklow = QLabel(self)
        self.stocksiga = QLabel(self)
        self.stockmachul = QLabel(self)
        self.stockyoungup = QLabel(self)
        self.per = QLabel(self)
        self.eps = QLabel(self)
        self.pbr = QLabel(self)
        self.bps = QLabel(self)

        self.stock = QPushButton("STOCKS")
        self.give = QPushButton("BINANCE FUTURES")
        self.btc_pre = QPushButton("BTC PREDICT")
        self.nas = QPushButton("OPTIONS")
        self.market = QPushButton("MARKETS")
        self.calin = QPushButton("CALLINDER")
        self.news = QPushButton("BROWSER")
        self.sltpcal = QPushButton("ORDERBOOK + SLIPPAGE")
        self.trv = QPushButton("Traiding View")
        self.kp = QPushButton("GIMP")

        self.stock.clicked.connect(self.stocks)
        self.give.clicked.connect(self.gives)
        self.btc_pre.clicked.connect(self.coin_pred)
        self.nas.clicked.connect(self.market_chart)
        self.market.clicked.connect(self.market_data)
        self.news.clicked.connect(self.news_crawl)
        self.sltpcal.clicked.connect(self.calculator)
        self.calin.clicked.connect(self.calindar_a)
        self.trv.clicked.connect(self.traiding_view)
        self.kp.clicked.connect(self.kimchi_pre)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        middleLayout = QVBoxLayout()
        middleLayout.addWidget(self.stockname)
        middleLayout.addWidget(self.stocksector)
        middleLayout.addWidget(self.stockindustry)
        middleLayout.addWidget(self.stockprice)
        middleLayout.addWidget(self.stockchigh)
        middleLayout.addWidget(self.stocklow)
        middleLayout.addWidget(self.stocksiga)
        middleLayout.addWidget(self.stockmachul)
        middleLayout.addWidget(self.stockyoungup)
        middleLayout.addWidget(self.per)
        middleLayout.addWidget(self.eps)
        middleLayout.addWidget(self.pbr)
        middleLayout.addWidget(self.bps)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.stock)
        rightLayout.addWidget(self.give)
        rightLayout.addWidget(self.btc_pre)
        rightLayout.addWidget(self.nas)
        rightLayout.addWidget(self.market)
        rightLayout.addWidget(self.calin)
        rightLayout.addWidget(self.news)
        rightLayout.addWidget(self.sltpcal)
        rightLayout.addWidget(self.trv)
        rightLayout.addWidget(self.kp)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(middleLayout)
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def stocks(self):
        try:
            text, ok = QInputDialog.getText(self, '주식', '종목명을 입력하세요')
            if ok:
                code = text.upper()
            dart = OpenDartReader("187387a3023799ce97c2dbe30b8f18e1db8ff17a")

            krx_mar=fdr.StockListing('KRX-MARCAP')
            dataset = krx_mar[krx_mar['Name']==code]

            df_krx = fdr.StockListing('krx')
            krx_dataset = df_krx[df_krx['Name']==code]

            symbol = str(krx_dataset.iloc[0,0])
            sec = krx_dataset.iloc[-1]['Sector']

            x = int(len(sec)/15)+1
            k=0
            sector = str()
            while k < x:
                a = sec[0+k*15:15+k*15]
                sector = sector+a+"\n"
                k = k + 1
            sector = sector+sec[15+(k-1)*15:]
            sanup = krx_dataset.iloc[-1]['Industry']
            x = int(len(sanup)/15)+1
            k=0
            industry = str()
            while k < x:
                a = sanup[0+k*15:15+k*15]
                industry = industry+a+"\n"
                k = k + 1
            industry = industry+sanup[15+(k-1)*15:]

            df = fdr.DataReader(symbol,str(datetime.today().year-1))
            high = max(df['High'])
            a = 0
            low_pr = []
            while a < len(df['Close']):
                if df.iloc[a]['Low'] == 0:
                    pass
                else:
                    low_pr.append(df.iloc[a]['Low'])
                a=a+1
            low = int(min(low_pr))
            stock_price = int(dataset['Close'])
            stock_amount = int(dataset['Stocks'])
            marcap = int(dataset['Marcap'])
            finance = dart.finstate(symbol, 2021, reprt_code='11011')
            machul = finance.iloc[9]['thstrm_amount']
            youngup = finance.iloc[10]['thstrm_amount']
            suniik = int(finance.iloc[12]['thstrm_amount'].replace(",", ""))
            jabon = int(finance.iloc[8]['thstrm_amount'].replace(",", ""))
            EPS = suniik/stock_amount
            PER = stock_price / EPS
            BPS = jabon/stock_amount
            PBR = stock_price / BPS
            siga = int(marcap/100000000)

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.plot(df.index, df['Close'], label="price")
            ax.legend(loc='upper right')
            ax.grid()
            self.canvas.draw()
            self.stockname.setText(code)
            self.stocksector.setText(sector)
            self.stockindustry.setText(industry)
            self.stockprice.setText("주가 : " + str(stock_price) + "원")
            self.stockchigh.setText("1년 최고가 : " + str(high) + "원")
            self.stocklow.setText("1년 최저가 : " + str(low) + "원")
            self.stocksiga.setText("시가총액 : " + str(siga) + "억원")
            self.stockmachul.setText("매출액 : " + str(machul) + "원")
            self.stockyoungup.setText("영업이익 : " + str(youngup) + "원")
            self.per.setText("PER : " + str(PER))
            self.eps.setText("EPS : " + str(EPS))
            self.pbr.setText("PBR : " + str(PBR))
            self.bps.setText("BPS : " + str(BPS))
        except Exception as e:
            self.stockname.setText("없는 종목입니다")
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")
    
    def gives(self):
        try:
            self.stockname.setText(" ")
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")
            bitcoin = 0
            long=[]
            short=[]
            binance = ccxt.binance()
            def get_symbols():
                usdt_markets = pd.DataFrame()
                frame = []
                markets = binance.load_markets()
                for ticker in markets:
                    name, fiat = ticker.split('/')
                    if fiat == 'USDT':
                        frame.append(ticker)
                usdt_markets['name'] = frame
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('UP')]
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('DOWN')]
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('BULL')]
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('BEAR')]
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('USDC')]
                usdt_markets = usdt_markets[~usdt_markets['name'].str.contains('UST')]
                usdt_markets.reset_index(drop=True, inplace=True)
                return usdt_markets['name']

            x = 2
            now = datetime.today()
            spots = get_symbols()
            for a in spots:
                btc_ohlcv = binance.fetch_ohlcv(symbol=a, timeframe="1d", limit=50)
                df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df.set_index('datetime', inplace=True)

                df['ma20'] = df['close'].rolling(window=20).mean()
                df['stddev'] = df['close'].rolling(window=20).std()
                df['upper'] = df['ma20'] + 2*df['stddev']
                df['lower'] = df['ma20'] - 2*df['stddev']

                if len(df) > 30 and int(df.index[-1].day) == int(now.day)-1:
                    if df.iloc[-x]['lower'] > df.iloc[-x]['low'] and df.iloc[-x]['lower'] < df.iloc[-x]['close']:
                        if df.iloc[-x]['volume'] > df.iloc[-x-1]['volume']*1.5:
                            bitcoin = bitcoin + 1
                        if df.iloc[-x]['volume'] < df.iloc[-x-1]['volume']:
                            long.append(a)
                    elif df.iloc[-x]['upper'] > df.iloc[-x]['close'] and df.iloc[-x]['upper'] < df.iloc[-x]['high']:
                        if df.iloc[-x]['volume'] > df.iloc[-x-1]['volume']*1.5:
                            bitcoin = bitcoin - 1
                        if df.iloc[-x]['volume'] < df.iloc[-x-1]['volume']:
                            short.append(a)
                    if len(long) > 9 or len(short) > 9:
                        break

            if bitcoin > 0:
                rc = ""
                for k in long:
                    rc = str(rc) +str("\n") + str(k)
                rate = "Recommand Position : Long"
            elif bitcoin < 0:
                rc = ""
                for k in short:
                    rc = str(rc) +str("\n") + str(k)
                rate = "Recommand Position : Short"
            else:
                rc = "보류"
                rate = "Hold"

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            btc_ohlcv2 = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe="1d", limit=100)
            btc = pd.DataFrame(btc_ohlcv2, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            btc['datetime'] = pd.to_datetime(btc['datetime'], unit='ms')
            btc.set_index('datetime', inplace=True)
            ax.plot(btc.index, btc['close'], label = rate)
            ax.legend(loc='upper right')
            colorset = mpf.make_marketcolors(up='tab:green', down='tab:red')
            s = mpf.make_mpf_style(marketcolors=colorset)
            mpf.plot(btc, type='candle', volume=True, style=s)
            self.canvas.draw()
            self.stockname.setText("Recommended Altcoins")
            self.stocksector.setText(rc)
        except Exception as e:
            self.stockname.setText(str(e))
    
    def coin_pred(self):
        self.fig.clear()
        self.stockname.setText(" ")
        self.stocksector.setText(" ")
        self.stockindustry.setText(" ")
        self.stockprice.setText(" ")            
        self.stockchigh.setText(" ")
        self.stocklow.setText(" ")
        self.stocksiga.setText(" ")
        self.stockmachul.setText(" ")
        self.stockyoungup.setText(" ")
        self.per.setText(" ")
        self.eps.setText(" ")
        self.pbr.setText(" ")
        self.bps.setText(" ")

        choose = {"ProChart","Pure"}
        c,ok = QInputDialog.getItem(self, "선물 가격 예측", "BINANCE FUTURES", choose)
            
        binance = ccxt.binance()
        result = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        x = 25
        while x > 0:
            now = datetime.today() - timedelta(hours=1000*x)
            since = round(now.timestamp()*1000)

            btc = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", since=since, limit=1000)
            data = pd.DataFrame(btc, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            data['datetime'] = pd.to_datetime(data['datetime'], unit='ms')
            data.set_index('datetime', inplace=True)
            x=x-1
            result = pd.concat([result,data])
        result = result.drop(['datetime'], axis='columns')  #파이썬 시간 +9 = 우리나라 시간

        btc_ohlcv = binance.fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=101)
        com = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        com['datetime'] = pd.to_datetime(com['datetime'], unit='ms')
        com.set_index('datetime', inplace=True)
        close = com['close']
        close = close[:100]

        base = (close - np.min(close)) / (np.max(close) - np.min(close))
        w = len(base)
        move = len(result) - w - 101
        sim = []
        for i in range(move):
            t = result.iloc[i:i+w]['close']
            base2 = (t - np.min(t)) / (np.max(t) - np.min(t))
            a = np.dot(base, base2) / (np.sqrt(np.dot(base ,base)) * np.sqrt(np.dot(base2, base2)))
            sim.append(a)

        ser = pd.Series(sim).sort_values(ascending = False).head(1)
        i = ser.index[0]
        chart = result.iloc[i:i+200]
        ta = []
        gap = com.iloc[-2]['close']/chart.iloc[100]['open']
        chart = chart * gap
        chart['volume'] = chart['volume'] / gap
        def get_chart(c):
            ta[0:100] = com[c]
            ta[100:] = chart.iloc[100:][c]
            chart[c] = ta
            return chart
        get_chart("close")
        get_chart("open")
        get_chart("high")
        get_chart("low")
        get_chart("volume")

        if c == "ProChart":
            df = chart
            df['ma20'] = df['close'].rolling(window=20).mean() # 20일 이동평균
            df['stddev'] = df['close'].rolling(window=20).std() # 20일 이동표준편차
            df['upper'] = df['ma20'] + 2*df['stddev'] # 상단밴드
            df['lower'] = df['ma20'] - 2*df['stddev'] # 하단밴드
            df['ma12'] = df['close'].rolling(window=12).mean() # 12일 이동평균
            df['ma26'] = df['close'].rolling(window=26).mean() # 26일 이동평균
            df['MACD'] = df['ma12'] - df['ma26']    # MACD
            df['MACD_Signal'] = df['MACD'].rolling(window=9).mean() # MACD Signal(MACD 9일 이동평균)
            df['MACD_Oscil'] = df['MACD'] - df['MACD_Signal']   #MACD 오실레이터
            df['ndays_high'] = df['high'].rolling(window=14, min_periods=1).max()    # 14일 중 최고가
            df['ndays_low'] = df['low'].rolling(window=14, min_periods=1).min()      # 14일 중 최저가
            df['fast_k'] = (df['close'] - df['ndays_low']) / (df['ndays_high'] - df['ndays_low']) * 100  # Fast %K 구하기
            df['slow_d'] = df['fast_k'].rolling(window=3).mean()    # Slow %D 구하기
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
            df['MFR'] = (df.PMF.rolling(window=10).sum() /
                df.NMF.rolling(window=10).sum())
            df['MFI10'] = 100 - 100 / (1 + df['MFR'])
            U = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
            D = np.where(df['close'].diff(1) < 0, df['close'].diff(1) *(-1), 0)
            AU = pd.DataFrame(U, index=df.index).rolling(window=14).mean()
            AD = pd.DataFrame(D, index=df.index).rolling(window=14).mean()
            RSI = AU / (AD+AU) *100
            df['RSI'] = RSI
            df = df[25:]

            candle = go.Candlestick(open=df['open'],high=df['high'],low=df['low'],close=df['close'], increasing_line_color = 'green',decreasing_line_color = 'red', showlegend=False)
            upper = go.Scatter( y=df['upper'], line=dict(color='royalblue', width=2), name='upper')
            ma20 = go.Scatter( y=df['ma20'], line=dict(color='black', width=2), name='ma20')
            lower = go.Scatter( y=df['lower'], line=dict(color='royalblue', width=2), name='lower')
            volume = go.Bar( y=df['volume'], marker_color='red', name='volume', showlegend=False)
            MACD = go.Scatter( y=df['MACD'], line=dict(color='blue', width=2), name='MACD')
            MACD_Signal = go.Scatter( y=df['MACD_Signal'], line=dict(dash='dashdot', color='green', width=2), name='MACD_Signal')
            MACD_Oscil = go.Bar( y=df['MACD_Oscil'], marker_color='purple', name='MACD_Oscil')
            fast_k = go.Scatter( y=df['fast_k'], line=dict(color='skyblue', width=2), name='fast_k')
            slow_d = go.Scatter( y=df['slow_d'], line=dict(dash='dashdot', color='black', width=2), name='slow_d')
            PB = go.Scatter( y=df['PB']*100, line=dict(color='blue', width=2), name='PB')
            MFI10 = go.Scatter( y=df['MFI10'], line=dict(dash='dashdot', color='green', width=2), name='MFI10')
            RSI = go.Scatter( y=df['RSI'], line=dict(color='red', width=2), name='RSI')
            fig = ms.make_subplots(rows=4, cols=2, specs=[[{'rowspan':3},{}],[None,{}],[None,{}],[{},{}]], shared_xaxes=True, horizontal_spacing=0.03, vertical_spacing=0.01)

            fig.add_trace(candle,row=1,col=1)
            fig.add_trace(upper,row=1,col=1)
            fig.add_trace(ma20,row=1,col=1)
            fig.add_trace(lower,row=1,col=1)
            fig.add_trace(volume,row=4,col=1)
            fig.add_trace(MACD,row=1,col=2)
            fig.add_trace(MACD_Signal,row=1,col=2)
            fig.add_trace(MACD_Oscil,row=1,col=2)
            fig.add_trace(fast_k,row=2,col=2)
            fig.add_trace(slow_d,row=2,col=2)
            fig.add_trace(PB,row=3,col=2)
            fig.add_trace(MFI10,row=3,col=2)
            fig.add_trace(RSI,row=4,col=2)
            fig.update_layout(shapes=[dict(x0=75,x1=75,y0=0,y1=1,xref='x',yref='paper',line_width=1)],annotations=[dict(x=22,y=1,xref='x',yref='paper',font=dict(color="black",size=30),showarrow=False,xanchor='left',text='<--present : future-->')])
            fig.update_layout(autosize=True, xaxis1_rangeslider_visible=False, xaxis2_rangeslider_visible=False, margin=dict(l=50,r=50,t=50,b=50), template='seaborn', title='BTC in next 100 hours - HongSeunguk')
            fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2, gridcolor='lightgray', showline=True,linewidth=2, linecolor='black', mirror=True)
            fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black', showgrid=True, gridwidth=2, gridcolor='lightgray',showline=True,linewidth=2, linecolor='black', mirror=True)
            fig.show()
        elif c == "Pure":
            df = chart[100:]
            ax = self.fig.add_subplot(111)
            ax.plot(df['close'], label = "BTC for the next 100 hours")
            ax.legend(loc='upper right')
            colorset = mpf.make_marketcolors(up='tab:green', down='tab:red')
            s = mpf.make_mpf_style(marketcolors=colorset)
            mpf.plot(df, type='candle', volume=True, style=s)
            self.canvas.draw()

    def market_chart(self):
        try:
            self.fig.clear()
            data_source = " "
            choose = {"예","아니오"}
            c,ok = QInputDialog.getItem(self, "지수차트", "상품 표를 조회하시겠습니까?", choose)
            if c == "예":
                self.stockname.setText("한국주요지수\n(KS11:KOSPI), (KQ11:KOSDAQ), (KS5:KOSPI50), (KS100:KOSPI100), (KRX100:KRX100), (KS200:KOSPI200)")
                self.stocksector.setText("미국주요지수\n(DJI:다우존스), (IXIC:나스닥종합), (US500:S&P 500),(RUTNU:러셀2000), (VIX:VIX공포지수)")
                self.stockindustry.setText("국가별대표지수\n(JP225:닛케이225), (STOXX50:유럽STOXX50), (HK50:항셍지수), (CSI300:중국CSI300), (TWII:대만가권지수), \n(HNX30:하노이종합), (SSEC:상해종합), (UK100:영국FTSE), (DE30:독일DAX30),(FCHI:프랑스CAC40)")
                self.stockprice.setText("상품 선물\n(NG:천연가스 선물), (GC:금 선물), (SI:은 선물), (HG:구리 선물), (CL:WTI유 선물)")
                self.stockchigh.setText("환률\n(USD/KRW:달러당 원화 환율), (USD/EUR:달러당 유로화 환율), (USD/JPY:달러당 엔화 환율), (CNY/KRW:위엔화 원화 환율), \n(EUR/USD:유로화 달러 환율), (USD/JPY:달러 엔화 환율), (JPY/KRW:엔화 원화 환율), (AUD/USD:오스트레일리아 달러 환율), \n(EUR/JPY:유로화 엔화 환율), (USD/RUB:달러 루블화")
                self.stocklow.setText("한국국채\n(KR1YT=RR:1년만기 한국 국채 수익률), (KR3YT=RR:3년만기 한국 국채 수익률), \n(KR5YT=RR:5년만기 한국 국채 수익률), (KR10YT=RR:10년만기 한국 국채 수익률)")
                self.stocksiga.setText("미국국채\n(US1MT=X:1개월 미국 국채 수익률), (US6MT=X:6개월 미국 국채 수익률), (US1YT=X:1년만기 미국 국채 수익률), \n(US5YT=X:5년만기 미국 국채 수익률), (US10YT=X:!0년만기 미국 국채 수익률), (US30YT=X:30년만기 미국 국채 수익률)")
                self.stockmachul.setText("암호화폐\n(BTC/KRW:비트코인 원화 가격), (ETH/KRW:이더리움 원화 가격), (XRP/KRW:리플 원화 가격), \n(BCH/KRW:비트코인 캐시 원화 가격), (EOS/KRW:이오스 원화 가격), (LTC/KRW:라이트 코인 원화 가격), \n(XLM/KRW:스텔라 원화 가격")
                self.stockyoungup.setText("BTC/USD:비트코인 달러 가격), (ETH/USD:이더리움 달러 가격), (XRP/USD:리플 달러 가격), \n(BCH/USD:비트코인 캐시 달러 가격), (EOS/USD:이오스 달러 가격), (LTC/USD:라이트 코인 달러 가격), \n(XLM/USD:스텔라 달러 가격)")
                self.per.setText("미국 연방 준비 은행 경제 지표 데이터\n(ICSA:주간 실업수당 청구 건수), (CCSA:연속 실업수당청구 건수), (UMCSENT:소비자심리지수), \n(HSN1F:주택 판매 지수), (UNRATE:실업률), (M2SL:M2 통화량), (BAMLH0A0HYM2:하이일드 채권 스프레드)")
                self.eps.setText(" ")
                self.pbr.setText(" ")
                self.bps.setText(" ")
            items = {"한국 주요 지수","미국 주요 지수","국가별 대표 지수","상품 선물","환율","한국국채","미국국채","암호화폐가격","FRED데이터"}
            x, ok = QInputDialog.getItem(self, "지수차트", "조회가능 상품", items)
            if x == "한국 주요 지수":
                mannuals = {"KS11","KQ11","KS50","KS100","KRX100","KS200"}
                y, ok = QInputDialog.getItem(self, "한국 주요 지수", "조회가능 상품", mannuals)
            elif x == "미국 주요 지수":
                mannuals = {"DJI","IXIC","US500","RUTNU","VIX"}
                y, ok = QInputDialog.getItem(self, "미국 주요 지수", "조회가능 상품", mannuals)
            elif x == "국가별 대표 지수":
                mannuals = {"JP225","STOXX50","HK50","CSI300","TWII","HNX30","SSEC","UK100","DE30","FCHI"}
                y, ok = QInputDialog.getItem(self, "국가별 대표 지수", "조회가능 상품", mannuals)
            elif x == "상품 선물":
                mannuals = {"NG","GC","SI","HG","CL"}
                y, ok = QInputDialog.getItem(self, "상품 선물", "조회가능 상품", mannuals)
            elif x == "환율":
                mannuals = {"USD/KRW","USD/EUR","USD/JPY","CNY/KRW","EUR/USD","USD/JPY","JPY/KRW","AUD/USD","EUR/JPY","USD/RUB"}
                y, ok = QInputDialog.getItem(self, "환율", "조회가능 상품", mannuals)
            elif x == "한국국채":
                mannuals = {"KR1YT=RR","KR3YT=RR","KR5YT=RR","KR10YT=RR","KR20YT=RR"}
                y, ok = QInputDialog.getItem(self, "한국 채권", "조회가능 상품", mannuals)
            elif x == "미국국채":
                mannuals = {"US1MT=X","US6MT=X","US1YT=X","US5YT=X","US10YT=X","US30YT=X"}
                y, ok = QInputDialog.getItem(self, "미국 채권", "조회가능 상품", mannuals)
            elif x == "암호화폐가격":
                mannuals = {"BTC/KRW","BTC/USD","ETH/KRW","ETH/USD","XRP/KRW","XRP/USD","BCH/KRW","BCH/USD","EOS/KRW","EOS/USD","LTC/KRW","LTC/USD","XLM/KRW","XLM/USD"}
                y, ok = QInputDialog.getItem(self, "암호화폐 가격", "조회가능 상품", mannuals)
            elif x == "FRED데이터":
                mannuals = {"ICSA","CCSA","UMCSENT","HSN1F","UNRATE","M2SL","BAMLH0A0HYM2"}
                y, ok = QInputDialog.getItem(self, "미국 연방 준비 은행 데이터", "조회가능 상품", mannuals)
                data_source = "fred"
            df = fdr.DataReader(y, str(datetime.today().year-1), data_source=data_source)
            ax = self.fig.add_subplot(111)
            if data_source == "fred":
                ax.plot(df.index,df[y])
                cr = float(df.iloc[-1])/float(df.iloc[-2])
                now_price = df.iloc[-1][y]
            else:
                colorset = mpf.make_marketcolors(up='tab:red', down='tab:blue', volume='tab:blue')
                s = mpf.make_mpf_style(marketcolors=colorset)
                ax.plot(df.index,df['Close'])
                mpf.plot(df, type='candle', style=s)
                cr = df.iloc[-1]['Change'] * 100
                now_price = df.iloc[-1]['Close']
            ax.legend(loc='upper right')
            ax.grid()
            self.canvas.draw()
            self.stockname.setText(str(y))
            self.stocksector.setText("현재가 : "+str(now_price))
            self.stockindustry.setText("변동률 : "+str(cr)+"%")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")
        except Exception as e:
            self.stockname.setText(str(e))
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")

    def market_data(self):
        try:
            self.fig.clear()
            def h_searching(url, sel):
                re=[]
                headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.48496.75" }
                original_html = requests.get(url, headers=headers)
                html = BeautifulSoup(original_html.text, "html.parser")
                a = html.select(sel)
                for x in a:
                    re.append(str(x.get_text()))
                return re
            kospi = str(h_searching("https://kr.investing.com/markets/south-korea", "tr#pair_37426 > td#last_37426"))
            kospi_var = str(h_searching("https://kr.investing.com/markets/south-korea", "tr#pair_37426 > td#chg_percent_37426"))
            kosdaq = str(h_searching("https://kr.investing.com/markets/south-korea", "tr#pair_38016 > td#last_38016"))
            kosdaq_var = str(h_searching("https://kr.investing.com/markets/south-korea", "tr#pair_38016 > td#chg_percent_38016"))
            kospi_tt = str(h_searching("https://finance.naver.com/sise/sise_index.naver?code=KOSPI", "dl.lst_kos_info > dd.dd"))
            kosdaq_tt = str(h_searching("https://finance.naver.com/sise/sise_index.naver?code=KOSDAQ", "dl.lst_kos_info > dd.dd"))
            nasdaq = str(h_searching("https://kr.investing.com/indices/nasdaq-composite", "span.text-2xl"))
            dow = str(h_searching("https://kr.investing.com/indices/us-30", "#__next > div.desktop\:relative.desktop\:bg-background-default > div > div > div.grid.gap-4.tablet\:gap-6.grid-cols-4.tablet\:grid-cols-8.desktop\:grid-cols-12.grid-container--fixed-desktop.general-layout_main__3tg3t > main > div > div.instrument-header_instrument-header__1SRl8.mb-5.bg-background-surface.tablet\:grid.tablet\:grid-cols-2 > div:nth-child(2) > div.instrument-price_instrument-price__3uw25.flex.items-end.flex-wrap.font-bold > span"))
            snp = str(h_searching("https://kr.investing.com/indices/us-spx-500", "#__next > div.desktop\:relative.desktop\:bg-background-default > div > div > div.grid.gap-4.tablet\:gap-6.grid-cols-4.tablet\:grid-cols-8.desktop\:grid-cols-12.grid-container--fixed-desktop.general-layout_main__3tg3t > main > div > div.instrument-header_instrument-header__1SRl8.mb-5.bg-background-surface.tablet\:grid.tablet\:grid-cols-2 > div:nth-child(2) > div.instrument-price_instrument-price__3uw25.flex.items-end.flex-wrap.font-bold > span"))
            news_e1 = str(h_searching("https://kr.investing.com/news/economy","#leftColumn > div.largeTitle > article:nth-child(1) > div.textDiv > a"))
            news_i1 = str(h_searching("https://kr.investing.com/news/economic-indicators", "#leftColumn > div.largeTitle > article:nth-child(1) > div.textDiv > a"))
            news_i2 = str(h_searching("https://kr.investing.com/news/economic-indicators", "#leftColumn > div.largeTitle > article:nth-child(2) > div.textDiv > a"))
            news_i3 = str(h_searching("https://kr.investing.com/news/economic-indicators", "#leftColumn > div.largeTitle > article:nth-child(3) > div.textDiv > a"))
            news_f1 = str(h_searching("https://kr.investing.com/news/forex-news", "#leftColumn > div.largeTitle > article:nth-child(1) > div.textDiv > a"))
            news_f2 = str(h_searching("https://kr.investing.com/news/forex-news", "#leftColumn > div.largeTitle > article:nth-child(2) > div.textDiv > a"))
            news_c1 = str(h_searching("https://kr.investing.com/news/cryptocurrency-news", "#leftColumn > div.largeTitle > article:nth-child(1) > div.textDiv > a"))
            a = fdr.DataReader("IXIC", str(datetime.today().year-1))
            nasdaq_tt = round((a.iloc[-1]['Close']/a.iloc[-2]['Close']-1)*100,2)
            b = fdr.DataReader("DJI", str(datetime.today().year-1))
            dow_tt = round((b.iloc[-1]['Close']/b.iloc[-2]['Close']-1)*100,2)
            c = fdr.DataReader("US500", str(datetime.today().year-1))
            snp_tt = round((c.iloc[-1]['Close']/c.iloc[-2]['Close']-1)*100,2)
            self.stockname.setText("코스피"+kospi+kospi_var+"\n"+kospi_tt+"\n")
            self.stocksector.setText("코스닥"+kosdaq+kosdaq_var+"\n"+kosdaq_tt+"\n")
            self.stockindustry.setText("다우 존스"+dow+str(dow_tt)+"%\n")
            self.stockprice.setText("나스닥 종합"+nasdaq+str(nasdaq_tt)+"%\n")           
            self.stockchigh.setText("s&p500"+snp+str(snp_tt)+"%\n")
            self.stocklow.setText("      ************뉴스************      ")
            self.stocksiga.setText(news_e1+"\n")
            self.stockmachul.setText(news_i1+"\n")
            self.stockyoungup.setText(news_i2+"\n")
            self.per.setText(news_i3+"\n")
            self.eps.setText(news_f1+"\n")
            self.pbr.setText(news_f2+"\n")
            self.bps.setText(news_c1)
        except Exception as e:
            self.stockname.setText(str(e))
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")

    def news_crawl(self):
        self.fig.clear()
        search, ok = QInputDialog.getText(self,"뉴스 검색","검색어를 입력하세요 : ")
        news_title = []
        url = []
        url.append("https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(1))
        url.append("https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(11))
        url.append("https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(21))
        a=0
        while a < 3:
            headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.48496.75" }
            original_html = requests.get(url[a], headers=headers)
            html = BeautifulSoup(original_html.text, "html.parser")
            articles = html.select("div.group_news > ul.list_news > li div.news_area > a")
            for i in articles:
                news_title.append(i.attrs['title'])
            a=a+1
        self.stockname.setText(news_title[0]+"   |   "+news_title[1])
        self.stocksector.setText(news_title[2]+"   |   "+news_title[3])
        self.stockindustry.setText(news_title[4]+"   |   "+news_title[5])
        self.stockprice.setText(news_title[6]+"   |   "+news_title[7])
        self.stockchigh.setText(news_title[8]+"   |   "+news_title[9])
        self.stocklow.setText(news_title[10]+"   |   "+news_title[11])
        self.stocksiga.setText(news_title[12]+"   |   "+news_title[13])
        self.stockmachul.setText(news_title[14]+"   |   "+news_title[15])
        self.stockyoungup.setText(news_title[16]+"   |   "+news_title[17])
        self.per.setText(news_title[18]+"   |   "+news_title[19])
        self.eps.setText(news_title[20]+"   |   "+news_title[21])
        self.pbr.setText(news_title[22]+"   |   "+news_title[23])
        self.bps.setText(news_title[24]+"   |   "+news_title[25])

    def calculator(self):
        try:
            self.fig.clear()
            items = {"Long","Short"}
            text, ok = QInputDialog.getText(self, '호가창&슬리피지', '티커명을 입력하세요')
            if ok:
                ticker = text.upper()
            pos, ok = QInputDialog.getItem(self, "호가창&슬리피지", "포지션선택", items)
            amount, ok = QInputDialog.getDouble(self, '호가창&슬리피지', '총 금액',100000,decimals=0)
            
            exchange = ccxt.binance()
            orderbook = exchange.fetch_order_book(ticker)
            if pos == "Long":
                oderbuk = "asks"
            elif pos == "Short":
                oderbuk = "bids"
            hap = 0
            price = []
            many = []
            start = orderbook[oderbuk][0][0]
            for i in orderbook[oderbuk]:
                hap = hap + (float(i[0])*float(i[-1]))
                price.append(i[0])
                many.append(i[-1])
                if hap > amount:
                    last = float(i[0])
                    break
            x = 0
            if len(price)==0:
                price.append("호가창 물량 충분함 : 예상 슬리피지 0%")
            lenth = 6-len(price)
            while x < lenth:
                price.append(" ")
                many.append(" ")
                x=x+1
            s = (last/start-1)*100
            self.stockname.setText(str(price[0])+"   "+str(many[0]))
            self.stocksector.setText(str(price[1])+"   "+str(many[1]))
            self.stockindustry.setText(str(price[2])+"   "+str(many[2]))
            self.stockprice.setText(str(price[3])+"   "+str(many[3]))
            self.stockchigh.setText(str(price[4])+"   "+str(many[4]))
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText("예상 슬리피지")
            self.stockyoungup.setText(str(round(s,3))+"%")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")
        except Exception as e:
            self.stockname.setText(str(e))
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")

    def calindar_a(self):
        try:
            a=[]
            b=[]
            c=[]
            news_title = pd.DataFrame({'time':[],'country':[],'title':[]})
            url = "https://kr.investing.com/economic-calendar/"
            headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0.48496.75" }
            original_html = requests.get(url, headers=headers)
            html = BeautifulSoup(original_html.text, "html.parser")
            title = html.select("td.left.event > a")
            time = html.select("td.first.left.time.js-time")
            country = html.select("td.left.flagCur.noWrap")
            for x in time:
                a.append(str(x.get_text()))
            for x in country:
                b.append(str(x.get_text()))
            for x in title:
                c.append(str(x.get_text()))
            x = 0
            while x < len(a):
                if a[x] == "잠정적인":
                    pass
                elif int(a[x][0:2]) > int(datetime.today().hour):
                    break
                x=x+1
            news_title['time'] = a[x:]
            news_title['country'] = b[x:]
            news_title['title'] = c[x:]
            x = 0
            price = []
            while x < len(news_title['time']):
                k = (str(news_title.iloc[x]['time'])+str(news_title.iloc[x]['country'])+" "+str(news_title.iloc[x]['title'])).replace("\n", "")
                price.append(k)
                x=x+1
            x = 0
            if len(price)==0:
                price.append("당일 일정이 없습니다")
            lenth = 26-len(price)
            while x < lenth:
                price.append(" ")
                x=x+1
            self.fig.clear()
            self.stockname.setText(price[0]+"  |  "+price[13])
            self.stocksector.setText(price[1]+"  |  "+price[14])
            self.stockindustry.setText(price[2]+"  |  "+price[15])
            self.stockprice.setText(price[3]+"  |  "+price[16])
            self.stockchigh.setText(price[4]+"  |  "+price[17])
            self.stocklow.setText(price[5]+"  |  "+price[18])
            self.stocksiga.setText(price[6]+"  |  "+price[19])
            self.stockmachul.setText(price[7]+"  |  "+price[20])
            self.stockyoungup.setText(price[8]+"  |  "+price[21])
            self.per.setText(price[9]+"  |  "+price[22])
            self.eps.setText(price[10]+"  |  "+price[23])
            self.pbr.setText(price[11]+"  |  "+price[24])
            self.bps.setText(price[12]+"  |  "+price[25])
        except Exception as e:
            self.stockname.setText(str(e))
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")

    def traiding_view(self):
        url = "https://kr.tradingview.com/chart/?symbol=BTCUSDT"
        webbrowser.open(url)

    def kimchi_pre(self):
        try:
            self.fig.clear()
            items = {"비트코인","이더리움","리플"}
            x, ok = QInputDialog.getItem(self, "김프 차트", "김프 차트", items)
            num, ok = QInputDialog.getInt(self, '김프 차트', '조회일수', 100, max=900)
            if x == "비트코인":
                ticker_k = "KRW-BTC"
                ticker_u = "BTC/BUSD"
            elif x == "이더리움":
                ticker_k = "KRW-ETH"
                ticker_u = "ETH/BUSD"
            elif x == "리플":
                ticker_k = "KRW-XRP"
                ticker_u = "XRP/BUSD"
            #btc-krw / (btc-usd * 환율)
            binance = ccxt.binance()
            btc_ohlcv = binance.fetch_ohlcv(symbol=ticker_u, timeframe="1d", limit=num)
            bu = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            bu['datetime'] = pd.to_datetime(bu['datetime'], unit='ms')
            bu.set_index('datetime', inplace=True)
            bk = pyupbit.get_ohlcv(ticker_k, interval = "day", count=num)
            fx = fdr.DataReader("USD/KRW", str(datetime.today().year-4))
            fx = fx[0:num]
            x=0
            kp = pd.DataFrame(index = bu.index, columns=['kp'])
            while x<num:
                kp.iloc[x]['kp'] = float(bk.iloc[x]['close'])/(float(bu.iloc[x]['close'])*float(fx.iloc[x]['Close']))
                x=x+1
            base = bk['close'] / ((np.max(bk['close']) - np.min(bk['close']))) * (np.max(kp['kp']) - np.min(kp['kp']))
            base = base - np.min(base) + np.min(kp['kp'])
            tt = (kp.iloc[-1]['kp']/kp.iloc[-2]['kp']-1)*100
            ax = self.fig.add_subplot(111)
            ax.plot(kp.index, kp['kp'], label="korean premium")
            ax.plot(kp.index, base, label="coin price")
            ax.legend(loc='upper right')
            ax.grid()
            self.canvas.draw()
            self.stockname.setText("현재가 : "+str(bk.iloc[-1]['close'])+"원")
            self.stocksector.setText(" ")
            self.stockindustry.setText("김프 : "+str(kp.iloc[-1]['kp'])+"%")
            self.stockprice.setText("김프 전일대비 변동률 : "+str(round(tt,4))+"%")          
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")
        except Exception as e:
            self.stockname.setText(str(e))
            self.stocksector.setText(" ")
            self.stockindustry.setText(" ")
            self.stockprice.setText(" ")            
            self.stockchigh.setText(" ")
            self.stocklow.setText(" ")
            self.stocksiga.setText(" ")
            self.stockmachul.setText(" ")
            self.stockyoungup.setText(" ")
            self.per.setText(" ")
            self.eps.setText(" ")
            self.pbr.setText(" ")
            self.bps.setText(" ")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
