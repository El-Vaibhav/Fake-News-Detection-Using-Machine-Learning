# ==============================
# FULL PLAGIARISM CHECKER
# ==============================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ddgs import DDGS
from bs4 import BeautifulSoup
from readability import Document
import requests
import nltk


# Load semantic model (lightweight + fast)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# TEXT PROCESSING
# ==============================

def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def compute_overall_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(similarity) * 100, 2)


def sentence_level_similarity(text1, text2, threshold=0.75):
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    if not sentences1 or not sentences2:
        return []

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    matches = []

    for i, row in enumerate(similarity_matrix):
        max_score = max(row)
        if max_score >= threshold:
            matched_index = row.argmax()
            matches.append({
                "input_sentence": sentences1[i],
                "matched_sentence": sentences2[matched_index],
                "similarity": round(float(max_score) * 100, 2)
            })

    return matches


# ==============================
# DUCKDUCKGO SEARCH
# ==============================

def search_duckduckgo(query, max_results=5):
    results_list = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

        for result in results:
            results_list.append({
                "title": result.get("title"),
                "link": result.get("href"),
                "snippet": result.get("body")
            })

    return results_list


# ==============================
# SCRAPER
# ==============================

def extract_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=5)

        doc = Document(response.text)
        html = doc.summary()

        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])

        return text.strip()

    except Exception as e:
        print("Scrape error:", e)
        return ""


# ==============================
# RISK LEVEL
# ==============================

def get_risk_level(score):
    if score > 75:
        return "High"
    elif score > 50:
        return "Moderate"
    else:
        return "Low"


# ==============================
# MAIN PLAGIARISM CHECKER
# ==============================

def check_web_similarity(user_text):

    query = user_text[:200]  # Use first 200 characters as search query

    print("\n🔎 Searching Web...\n")
    search_results = search_duckduckgo(query, max_results=5)

    report = []
    highest_score = 0

    for result in search_results:
        url = result["link"]
        title = result["title"]

        print(f"🌐 Checking: {url}")

        web_text = extract_text_from_url(url)

        if len(web_text) < 300:
            continue

        overall_score = compute_overall_similarity(user_text, web_text)
        sentence_matches = sentence_level_similarity(user_text, web_text)

        if overall_score > highest_score:
            highest_score = overall_score

        report.append({
            "title": title,
            "url": url,
            "similarity": overall_score,
            "matched_sentences": sentence_matches[:3]
        })

    report = sorted(report, key=lambda x: x["similarity"], reverse=True)

    return {
        "highest_similarity": highest_score,
        "risk_level": get_risk_level(highest_score),
        "sources": report
    }


# ==============================
# RUN TEST
# ==============================

if __name__ == "__main__":

    user_text = """
At a global summit, India is pushing the idea of a shared “AI commons” framework so countries can access interoperable AI technologies for education and healthcare.    """

    result = check_web_similarity(user_text)

    print("\n================ FINAL REPORT ================\n")

    print("Highest Similarity:", result["highest_similarity"], "%")
    print("Risk Level:", result["risk_level"])
    print("\nSources:\n")

    for source in result["sources"]:
        print("Title:", source["title"])
        print("URL:", source["url"])
        print("Similarity:", source["similarity"], "%")
        print("Matched Sentences:", source["matched_sentences"])
        print("-" * 80)
