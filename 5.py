import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

# 데이터 준비 함수
def get_ethereum_price_data(start_date, end_date, interval='1d'):
    eth = yf.Ticker("ETH-USD")
    data = eth.history(start=start_date, end=end_date, interval=interval)
    return data

# 날짜 범위 설정
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)

# 이더리움 가격 데이터 가져오기
eth_data = get_ethereum_price_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if eth_data.empty:
    print("데이터를 가져오는 데 실패했습니다.")
else:
    # 종가 데이터만 사용
    data = eth_data.filter(['Close'])
    dataset = data.values

    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 학습용 데이터 생성
    training_data_len = int(np.ceil(len(dataset) * .95))

    train_data = scaled_data[0:int(training_data_len), :]

    # 학습 데이터를 입력(X_train)와 출력(y_train)으로 분리
    X_train = []
    y_train = []

    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    # 배열을 numpy 배열로 변환
    X_train, y_train = np.array(X_train), np.array(y_train)

    # LSTM 모델 입력 형태에 맞게 데이터 재구성
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # LSTM 모델 생성
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # 테스트 데이터셋 생성
    test_data = scaled_data[training_data_len - 60:, :]

    # 테스트 데이터를 입력(X_test)와 출력(y_test)으로 분리
    X_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
        
    # 배열을 numpy 배열로 변환
    X_test = np.array(X_test)

    # LSTM 모델 입력 형태에 맞게 데이터 재구성
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 가격 예측
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # RMSE 계산
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f'Root Mean Squared Error: {rmse}')

    # 예측 데이터 시각화
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(12, 6))
    plt.title('Ethereum Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    # 실제 가격 (파란선)
    plt.plot(data['Close'], color='blue', label='Actual Price')
    # 예측 가격 (노란선)
    plt.plot(valid.index, valid['Predictions'], color='yellow', label='Predicted Price')
    plt.legend(loc='lower right')
    plt.show()
