import yfinance as yf
import matplotlib.pyplot as plt

# 이더리움 가격 데이터 다운로드
eth_data = yf.download('ETH-USD', start='2024-07-01', end='2024-07-30')

# 데이터 확인
print(eth_data.head())

# 이더리움 가격 시각화
plt.figure(figsize=(10, 6))
plt.plot(eth_data['Close'], label='ETH-USD')
plt.title('Ethereum Price (ETH-USD)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()