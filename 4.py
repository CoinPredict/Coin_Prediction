import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 준비 함수
def get_ethereum_price_data(start_date, end_date, interval='1d'):
    eth = yf.Ticker("ETH-USD")
    data = eth.history(start=start_date, end=end_date, interval=interval)
    return data

# 날짜 범위 설정
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=240)

# 이더리움 가격 데이터 가져오기
eth_data = get_ethereum_price_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if eth_data.empty:
    print("데이터를 가져오는 데 실패했습니다.")
else:
    # 데이터 준비
    eth_data['Date'] = eth_data.index
    eth_data['Date'] = pd.to_datetime(eth_data['Date'])
    eth_data['Date_ordinal'] = eth_data['Date'].map(pd.Timestamp.toordinal)
    eth_data['Volume_Mean'] = eth_data['Volume'].rolling(window=3).mean().fillna(eth_data['Volume'].mean())

    # 결측값 및 이상값 처리
    eth_data = eth_data.dropna()
    
    # 데이터 스케일링
    scaler = StandardScaler()
    eth_data[['Scaled_Close', 'Scaled_Volume_Mean']] = scaler.fit_transform(eth_data[['Close', 'Volume_Mean']])

    # 240일 데이터를 75% 학습용, 25% 예측용으로 나누기
    train_size = int(len(eth_data) * 0.75)
    train_data = eth_data[:train_size].copy()
    test_data = eth_data[train_size:].copy()

    # 랜덤 포레스트 모델 학습
    X_train = train_data[['Date_ordinal', 'Scaled_Volume_Mean']]
    y_train = train_data['Scaled_Close']
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10).fit(X_train, y_train)

    # 예측
    X_test = test_data[['Date_ordinal', 'Scaled_Volume_Mean']]
    test_data['Predicted_Scaled'] = model.predict(X_test)

    # 스케일링 복원
    test_data['Predicted'] = scaler.inverse_transform(np.array([test_data['Predicted_Scaled'], test_data['Scaled_Volume_Mean']]).T)[:, 0]

    # 예측값을 전체 데이터 프레임에 추가
    eth_data['Predicted'] = pd.Series([None]*len(eth_data), index=eth_data.index)
    eth_data.loc[test_data.index, 'Predicted'] = test_data['Predicted']

    # 30일 이동 평균과 60일 이동 평균 계산
    eth_data['MA30'] = eth_data['Close'].rolling(window=30).mean()
    eth_data['MA60'] = eth_data['Close'].rolling(window=60).mean()

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(eth_data[-60:]['Date'], eth_data[-60:]['Close'], label='ETH-USD Price')
    plt.plot(test_data['Date'], test_data['Predicted'], label='Predicted ETH-USD', linestyle='--', color='orange')
    plt.plot(eth_data[-60:]['Date'], eth_data[-60:]['MA30'], label='30-day MA', linestyle='--', color='red')
    plt.plot(eth_data[-60:]['Date'], eth_data[-60:]['MA60'], label='60-day MA', linestyle='--', color='brown')
    plt.title('Ethereum Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # 예측 정확도 평가 (예: MSE)
    mse = mean_squared_error(test_data['Close'], test_data['Predicted'])
    print(f'Mean Squared Error of the prediction: {mse}')
