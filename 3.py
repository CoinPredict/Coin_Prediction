import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import matplotlib.font_manager as fm


font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path, size=12)


# 이더리움 데이터 다운로드
def get_ethereum_price_data(period='1d', interval='1m'):
    # 이더리움의 Yahoo Finance 티커
    eth = yf.Ticker("ETH-USD")
    
    # 가격 데이터 다운로드
    data = eth.history(period=period, interval=interval)
    return data

# 가격 데이터를 가져오고 시각화
def plot_price_data():
    # 데이터 가져오기
    eth_data = get_ethereum_price_data()

    
    if not eth_data.empty:
        # 그래프 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(eth_data.index, eth_data['Close'], label='이더리움 가격', color='blue')
        plt.title('이더리움 분 단위 가격 변동', fontproperties=font_prop)
        plt.xlabel('시간', fontproperties=font_prop)
        plt.ylabel('가격 (USD)', fontproperties=font_prop)
        plt.xticks(rotation=45, fontproperties=font_prop)
        plt.legend(prop=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("데이터를 가져오는 데 실패했습니다.")

# 실행
plot_price_data()
