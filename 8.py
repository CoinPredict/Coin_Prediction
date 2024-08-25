import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

data = pd.read_csv('TSLA-2.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

mpf.plot(data, type='candle', volume=True, 
         title='Tesla Stock Candlestick Chart', 
         ylabel='Price (USD)', 
         ylabel_lower='Volume', 
         figsize=(14, 7))  # 그래프 크기 조정