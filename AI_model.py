import pandas as pd
import numpy as np
import re
import nltk
import joblib
import math

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = set(stopwords.words("english"))

# ====================================================
# 1. LOAD DATA
# ====================================================

ai_df = pd.read_csv("ai_generated_news_20k.csv")
real_df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\Fake_News_Detection\\Fake-News-Detection-Using-Machine-Learning\\True.csv")

ai_df = ai_df.rename(columns={"Article_Body": "text"})
ai_df["label"] = 1

real_df["text"] = real_df["title"].fillna("") + " " + real_df["text"].fillna("")
real_df["label"] = 0

ai_df = ai_df[["text", "label"]]
real_df = real_df[["text", "label"]]

df = pd.concat([ai_df, real_df]).reset_index(drop=True)

print("Original Dataset Size:", df.shape)
print(df["label"].value_counts())

# ====================================================
# 2. REMOVE DATASET BIAS & SOURCE LEAKAGE
# ====================================================

def remove_source_bias(text):
    text = text.lower()

    # Remove Reuters style signatures
    text = re.sub(r'\(reuters\)', '', text)
    text = re.sub(r'reuters', '', text)

    # Remove location prefixes like "WASHINGTON -"
    text = re.sub(r'^[A-Z\s]+-\s', '', text)

    # Remove common publishing patterns
    text = re.sub(r'by [a-z\s]+', '', text)
    text = re.sub(r'last updated.*', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["text"] = df["text"].apply(remove_source_bias)

# ====================================================
# 3. BALANCE DATASET
# ====================================================

min_count = df["label"].value_counts().min()

df = df.groupby("label").sample(min_count, random_state=42)
df = df.reset_index(drop=True)

print("Balanced Dataset Size:", df.shape)
print(df["label"].value_counts())

# ====================================================
# 4. STYLOMETRIC FEATURES
# ====================================================

def extract_stylometric_features(text):

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]

    if len(sentences) == 0 or len(words) == 0:
        return [0]*10

    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    burstiness = np.var(sentence_lengths)
    avg_sentence_length = np.mean(sentence_lengths)

    word_lengths = [len(w) for w in words]
    avg_word_length = np.mean(word_lengths)

    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words)

    stopword_ratio = sum(w in stop_words for w in words) / len(words)

    counts = Counter(words)
    repetition_score = max(counts.values()) / len(words)

    total_words = len(words)
    entropy = -sum((c/total_words) * math.log2(c/total_words)
                   for c in counts.values())

    hapax_ratio = len([w for w, c in counts.items() if c == 1]) / total_words

    return [
        burstiness,
        avg_sentence_length,
        avg_word_length,
        lexical_diversity,
        stopword_ratio,
        repetition_score,
        entropy,
        hapax_ratio,
        total_words,
        len(sentences)
    ]

print("Extracting stylometric features...")
stylometric_features = np.array(
    df["text"].apply(extract_stylometric_features).tolist()
)

# ====================================================
# 5. TF-IDF (REDUCED POWER TO PREVENT MEMORIZATION)
# ====================================================

tfidf = TfidfVectorizer(
    max_features=15000,        # reduced
    ngram_range=(1,1),         # unigrams only
    min_df=10,
    max_df=0.8
)

tfidf_matrix = tfidf.fit_transform(df["text"])

# ====================================================
# 6. SCALE STYLOMETRIC FEATURES
# ====================================================

scaler = StandardScaler()
stylometric_scaled = scaler.fit_transform(stylometric_features)

# ====================================================
# 7. COMBINE FEATURES
# ====================================================

X = hstack([tfidf_matrix, stylometric_scaled])
y = df["label"]

# ====================================================
# 8. MODEL WITH STRONGER REGULARIZATION
# ====================================================

model = LogisticRegression(
    max_iter=3000,
    C=0.5,                    # stronger regularization
    class_weight="balanced"
)

# ====================================================
# 9. CROSS VALIDATION (TRUE EVALUATION)
# ====================================================

print("\nRunning 5-Fold Cross Validation...")
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Accuracy:", cv_scores.mean())

# ====================================================
# 10. FINAL TRAIN TEST SPLIT
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ====================================================
# 11. SAVE SINGLE PICKLE
# ====================================================

complete_pipeline = {
    "model": model,
    "tfidf": tfidf,
    "scaler": scaler
}

joblib.dump(complete_pipeline, "ai_detector_realistic.pkl")

print("\nRealistic AI Detector saved as ai_detector_realistic.pkl")
