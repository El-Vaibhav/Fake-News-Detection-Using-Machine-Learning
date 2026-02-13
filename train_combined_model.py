import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------
# Preprocessing
# -----------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------
# Load Data
# -----------------------
fake = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\Fake_News_Detection\\Fake-News-Detection-Using-Machine-Learning\\Fake.csv")
real = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\Fake_News_Detection\\Fake-News-Detection-Using-Machine-Learning\\True.csv")

fake["label"] = 0
real["label"] = 1

# Combine title + text (IMPORTANT)
fake["content"] = fake["title"] + " " + fake["text"]
real["content"] = real["title"] + " " + real["text"]

data = pd.concat([fake[["content", "label"]], real[["content", "label"]]])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess
data["content"] = data["content"].apply(preprocess)

X = data["content"]
y = data["label"]

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# TF-IDF (STRONGER VERSION)
# -----------------------
tfidf = TfidfVectorizer(
    ngram_range=(1,3),        # Unigrams + Bigrams + Trigrams
    max_features=50000,
    min_df=3,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------
# Classifier (BETTER THAN BASIC LOGISTIC)
# -----------------------
model = SGDClassifier(
    loss="log_loss",          # logistic regression
    max_iter=3000,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------
# Save Model
# -----------------------
joblib.dump({
    "model": model,
    "tfidf": tfidf
}, "improved_fake_news_model.pkl", compress=3)

print("\n✅ Improved model saved successfully!")
