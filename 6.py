import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

# Cointelegraph에서 이더리움 관련 뉴스 데이터 크롤링
def fetch_news():
    url = 'https://cointelegraph.com/tags/ethereum'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    
    news_data = []
    for article in articles:
        title = article.find('h2').text.strip()
        time_tag = article.find('time')
        date = time_tag['datetime'] if time_tag and 'datetime' in time_tag.attrs else None
        news_data.append({'title': title, 'date': date})
    
    news_df = pd.DataFrame(news_data)
    
    # 'date' 열에서 None 값을 제외하고, 'date' 열이 존재할 경우에만 변환
    news_df = news_df.dropna(subset=['date'])
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    else:
        print("유효한 'date' 정보가 포함된 뉴스 기사가 없습니다.")
    return news_df

# 감성 분석 수행
def perform_sentiment_analysis(news_df):
    news_df['sentiment'] = news_df['title'].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
    return news_df

# 이더리움 가격 데이터 수집
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
    # 뉴스 데이터 크롤링 및 감성 분석
    news_df = fetch_news()
    if not news_df.empty:
        news_df = perform_sentiment_analysis(news_df)

        # 뉴스 데이터를 가격 데이터와 결합
        eth_data['date'] = eth_data.index.date
        merged_df = pd.merge(eth_data, news_df, on='date', how='left')
        merged_df = merged_df.fillna(0)

        # 종가 데이터와 감성 점수만 사용
        data = merged_df[['Close', 'sentiment']]
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
            X_train.append(train_data[i-60:i, :])
            y_train.append(train_data[i, 0])
            
        # 배열을 numpy 배열로 변환
        X_train, y_train = np.array(X_train), np.array(y_train)

        # LSTM 모델 입력 형태에 맞게 데이터 재구성
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        # LSTM 모델 생성
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
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
        y_test = dataset[training_data_len:, 0]
        for i in range(60, len(test_data)):
            X_test.append(test_data[i-60:i, :])
            
        # 배열을 numpy 배열로 변환
        X_test = np.array(X_test)

        # LSTM 모델 입력 형태에 맞게 데이터 재구성
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # 가격 예측
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]

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
    else:
        print("유효한 뉴스 데이터가 없어 감성 분석 및 모델 학습을 진행할 수 없습니다.")
