import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

# Cointelegraph URL 설정
url = 'https://cointelegraph.com/tags/ethereum'

# User-Agent 헤더 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 요청 보내기
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# HTML 구조 확인
print(soup.prettify())

# 뉴스 기사 추출
articles = soup.find_all('article')

# 데이터 저장을 위한 리스트
news_data = []

for article in articles:
    title_tag = article.find('h2')
    time_tag = article.find('time')
    
    if title_tag and time_tag:
        title = title_tag.text.strip()
        date = time_tag['datetime']
        news_data.append({'title': title, 'date': date})

# 데이터프레임으로 변환
news_df = pd.DataFrame(news_data)

# JSON 파일로 저장
news_df.to_json('./data1.json', orient='records', force_ascii=False, indent=4)

print(news_df.head())
