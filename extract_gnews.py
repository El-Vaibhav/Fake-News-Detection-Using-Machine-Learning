import requests
import pandas as pd
import time

# ===============================
# CONFIGURATION
# ===============================

API_KEY = "2f7fc9caeec6bfab63ad89d26bbf6c01"   # Replace with your key
BASE_URL = "https://gnews.io/api/v4/search"

QUERY = "politics OR economy OR technology"
LANGUAGE = "en"
MAX_ARTICLES = 2000  # total articles you want
ARTICLES_PER_REQUEST = 10  # free tier usually allows 10 max per request

# ===============================
# FUNCTION TO FETCH NEWS
# ===============================

def fetch_news():
    all_articles = []
    page = 1

    while len(all_articles) < MAX_ARTICLES:
        params = {
            "q": QUERY,
            "lang": LANGUAGE,
            "max": ARTICLES_PER_REQUEST,
            "page": page,
            "apikey": API_KEY
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("Error:", response.json())
            break

        data = response.json()

        articles = data.get("articles", [])

        if not articles:
            print("No more articles found.")
            break

        for article in articles:
            all_articles.append({
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "url": article.get("url"),
                "image": article.get("image"),
                "publishedAt": article.get("publishedAt"),
                "source_name": article.get("source", {}).get("name")
            })

        print(f"Fetched page {page} | Total collected: {len(all_articles)}")

        page += 1
        time.sleep(1)  # avoid rate limit

    return all_articles[:MAX_ARTICLES]


# ===============================
# SAVE TO CSV
# ===============================

def save_to_csv(articles):
    df = pd.DataFrame(articles)
    df.to_csv("gnews_articles.csv", index=False)
    print("Saved to gnews_articles.csv")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    articles = fetch_news()
    save_to_csv(articles)
