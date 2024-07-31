import requests
import datetime
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json

font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path, size=12)

API_KEY = "e9aa996fb9613bc66dbae88eb17bd7461d9f896f84eea8d7010465ac54b8e8d9"

def get_ethereum_price():
    url = f"https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD&api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    return data['USD']

def get_historical_ethereum_prices(days):
    to_ts = int(datetime.datetime.now().timestamp())
    limit = 2000
    historical_data = []
    total_minutes = days * 24 * 60
    
    for i in range(0, total_minutes, limit):
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym=ETH&tsym=USD&limit={min(limit, total_minutes - i)}&toTs={to_ts}&api_key={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if 'Data' in data and 'Data' in data['Data']:
            historical_data.extend(data['Data']['Data'])
            to_ts = data['Data']['Data'][-1]['time']
        else:
            messagebox.showerror("Error", "데이터를 가져오는 데 오류가 발생했습니다.")
            return []
        

    return historical_data

def show_price():
    price = get_ethereum_price()
    price_label.config(text=f"현재 이더리움 가격: ${price}")

def show_historical_prices():
    days = int(days_entry.get())
    historical_prices = get_historical_ethereum_prices(days)
    
    if historical_prices:
        timestamps = [datetime.datetime.fromtimestamp(price['time']) for price in historical_prices]
        prices = [price['close'] for price in historical_prices]

        # 그래프 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, prices, label='이더리움 가격', color='blue')
        plt.title(f'과거 {days}일간의 이더리움 가격', fontproperties=font_prop)
        plt.xlabel('날짜', fontproperties=font_prop)
        plt.ylabel('가격 (USD)', fontproperties=font_prop)
        plt.xticks(rotation=45, fontproperties=font_prop)
        plt.legend(prop=font_prop)
        plt.tight_layout()
        plt.show()

# GUI 설정
root = tk.Tk()
root.title("이더리움 가격 예측 프로그램")

# 현재 가격 버튼
current_price_button = tk.Button(root, text="현재 이더리움 가격 보기", command=show_price)
current_price_button.pack(pady=10)

price_label = tk.Label(root, text="")
price_label.pack(pady=10)

# 과거 가격 버튼
days_label = tk.Label(root, text="과거 가격 데이터 가져올 일수 입력:")
days_label.pack(pady=5)

days_entry = tk.Entry(root)
days_entry.pack(pady=5)

historical_price_button = tk.Button(root, text="과거 가격 보기", command=show_historical_prices)
historical_price_button.pack(pady=10)

root.mainloop()