import pandas as pd
import pickle
import re
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def get_word_count(text):
    return len(text.split())

# ----------------------------
# Load data
# ----------------------------
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["class"] = 0
real["class"] = 1

data = pd.concat([fake, real])
data = data.drop(["title", "subject", "date"], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

data["text"] = data["text"].apply(clean_text)
data["word_count"] = data["text"].apply(get_word_count)

X = data[["word_count"]]
y = data["class"]

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ----------------------------
# Save model
# ----------------------------
with open("fake_news_wordcount_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")
