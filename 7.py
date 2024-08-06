import requests
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def fetch_news(api_key, search_queries, total_articles=500):
    api_url = 'https://newsapi.org/v2/everything'
    articles = []
    
    for search_query in search_queries:
        pages = total_articles // 100 + (1 if total_articles % 100 != 0 else 0) 
        requests_made = 0  
        
        for page in range(1, pages + 1):
            params = {
                'q': search_query,  
                'apiKey': api_key,
                'pageSize': 100,
                'page': page,  
                'language': 'en'
            }

            response = requests.get(api_url, params=params)
            requests_made += 1 
            
            if response.status_code == 200:
                data = response.json()
                articles += data.get('articles', [])
            else:
                print(f"API 요청 실패: {response.status_code}")
                break

        print(f"{search_query} 검색 완료, 요청 횟수: {requests_made}")

    news_df = pd.DataFrame(articles)
    print(f"수집한 뉴스 기사 수: {len(news_df)}")
    return news_df[['title', 'publishedAt', 'url']]



def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    return text.lower() 


def sentiment_analysis(news_df):

    y = [1 if i % 2 == 0 else 0 for i in range(len(news_df))]  # 더미 데이터


    news_df['cleaned_title'] = news_df['title'].apply(preprocess_text)


    X = news_df['cleaned_title']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    model = MultinomialNB()
    model.fit(X_train_vec, y_train)


    y_pred = model.predict(X_test_vec)


    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy:.2f}")

def main():
    api_key = '0dc09f20733347919a67e056e9669dde' 
    search_queries = ['Trump', 'BTC', 'election', 'Harris', 'Biden', 'bubble', 'NASDAQ', 'VIX']  
    news_df = fetch_news(api_key, search_queries)

    if news_df is not None and not news_df.empty:
        print("수집한 뉴스 데이터:")
        print(news_df.head())
        sentiment_analysis(news_df)
    else:
        print("뉴스 데이터가 없습니다.")

if __name__ == "__main__":
    main()
