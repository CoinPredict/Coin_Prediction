import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import matplotlib.font_manager as fm
from statsmodels.graphics.tsaplots import plot_acf

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path, size=12)

def get_date_range(days):
    """지정된 일수만큼의 날짜 범위를 반환합니다."""
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    return start_date, end_date

def get_ethereum_price_data(start_date, end_date, interval='1d'):
    """주어진 날짜 범위에 대한 이더리움 가격 데이터를 가져옵니다."""
    eth = yf.Ticker("ETH-USD")
    data = eth.history(start=start_date, end=end_date, interval=interval)
    return data

def plot_moving_averages_with_range(days):
    """지정된 범위에서 이더리움 가격과 이동 평균을 시각화하고 자기상관성을 플롯합니다."""
    # 90일 전 데이터 범위 가져오기 (60일 이동 평균 계산을 위해 충분한 데이터 확보)
    start_date, end_date = get_date_range(days + 60)
    
    # 가격 데이터 가져오기
    eth_data = get_ethereum_price_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if eth_data.empty:
        print("데이터를 가져오는 데 실패했습니다.")
        return

    # 이동 평균 계산
    eth_data['MA30'] = eth_data['Close'].rolling(window=30).mean()
    eth_data['MA60'] = eth_data['Close'].rolling(window=60).mean()

    # 최근 30일 데이터만 선택
    recent_data = eth_data[-days:]

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(recent_data.index, recent_data['Close'], label='ETH-USD (Last 30 Days)', color='blue')
    plt.plot(recent_data.index, recent_data['MA30'], label='30-Day Moving Average', color='orange', linestyle='--')
    plt.plot(recent_data.index, recent_data['MA60'], label='60-Day Moving Average', color='green', linestyle='--')  # 전체 60일 이동 평균선
    plt.title('Ethereum Price with Moving Averages (Last {} Days)'.format(days), fontproperties=font_prop)
    plt.xlabel('Date', fontproperties=font_prop)
    plt.ylabel('Price (USD)', fontproperties=font_prop)
    plt.xticks(rotation=45, fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.tight_layout()
    plt.show()

    # 자기상관성 플롯
    lags = min(30, len(recent_data) - 1)  # lags 값 조정
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(recent_data['Close'], lags=lags, zero=False, ax=ax)
    ax.set_title('Autocorrelation of Ethereum Prices (Last {} Days)'.format(days), fontproperties=font_prop)
    ax.set_xlabel('Lag', fontproperties=font_prop)
    ax.set_ylabel('Autocorrelation', fontproperties=font_prop)
    
    # x축과 y축의 폰트를 설정
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    
    plt.show()

# 실행
if __name__ == "__main__":
    plot_moving_averages_with_range(days=30)  # 최근 30일간의 가격 데이터 및 이동 평균 시각화
