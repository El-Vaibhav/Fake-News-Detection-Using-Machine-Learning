import joblib
import re
import string
import numpy as np

# -----------------------
# Preprocessing (same as training)
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
# Load Model
# -----------------------
model_data = joblib.load("improved_fake_news_model.pkl")

model = model_data["model"]
tfidf = model_data["tfidf"]

print("✅ Model loaded successfully!\n")


# -----------------------
# Fake News Sample (Testing)
# -----------------------
sample_news = """
Delhi schools get fresh bomb threats; students, teachers evacuated
Delhi Fire Services (DFS) officials have said that authorities have launched search operations.

Several schools in Delhi received bomb threat emails on Friday morning, said police. Delhi Police, along with fire department and bomb disposal squads, were deployed to various school campuses to carry out extensive anti-sabotage checks.
On February 9, more than 15 schools received threats.

According to people aware of the developments, threats were sent to BT Tamil School in Jhandewalan, Sardar Patel Vidyalaya and the British School.

Police said the calls started coming in around 9:12am about the bomb threats from the three schools. Later, more schools informed police about a similar bomb threat email.

Senior police officers said the mail was sent using an anonymous mail ID.

In a message to parents, a school said, "Dear Parents, this morning the school received a security threat. As a precautionary measure, the police are in school for necessary security measures. All students have been evacuated safely. Once the school is declared safe, classes will be resumed".
"""

# -----------------------
# Prediction
# -----------------------
processed = preprocess(sample_news)
vector = tfidf.transform([processed])

prediction = model.predict(vector)[0]
probability = model.predict_proba(vector)[0]

fake_confidence = probability[0] * 100
real_confidence = probability[1] * 100

print("Prediction:", "🟥 FAKE" if prediction == 0 else "🟩 REAL")
print(f"Confidence FAKE: {fake_confidence:.2f}%")
print(f"Confidence REAL: {real_confidence:.2f}%")
