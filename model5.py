# Step 1: Import necessary libraries
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Step 2: Load Data
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# Adding a 'class' column: fake_news -> 0, real_news -> 1
fake_news["class"] = 0
real_news["class"] = 1

# Combine datasets
combined_data = pd.concat([fake_news, real_news], axis=0)

# Remove unnecessary columns
final_data = combined_data.drop(["title", "subject", "date"], axis=1)

# Shuffle the data
final_data = final_data.sample(frac=1).reset_index(drop=True)

# Step 3: Preprocess Data and Split into Train/Test
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Clean text data
final_data['text'] = final_data['text'].apply(wordopt)

# Define features and target
x = final_data['text']
y = final_data['class']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 4: Feature Extraction with TF-IDF, Sentiment Analysis, and POS Analysis
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Initialize Vader Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def calculate_sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

# Apply sentiment analysis to training and testing data
train_sentiment_scores = x_train.apply(calculate_sentiment_scores)
test_sentiment_scores = x_test.apply(calculate_sentiment_scores)

# Function to calculate POS features
def calculate_pos_features(text):
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    total_count = sum(pos_counts.values())
    pos_features = {
        'ADJ': pos_counts.get(nlp.vocab.strings['ADJ'], 0) / total_count,
        'NOUN': pos_counts.get(nlp.vocab.strings['NOUN'], 0) / total_count,
        'VERB': pos_counts.get(nlp.vocab.strings['VERB'], 0) / total_count,
        'ADV': pos_counts.get(nlp.vocab.strings['ADV'], 0) / total_count
    }
    return pd.Series(pos_features)

# Apply POS feature extraction to training and testing data
train_pos_features = x_train.apply(calculate_pos_features)
test_pos_features = x_test.apply(calculate_pos_features)

# Concatenate sentiment scores and POS features with TF-IDF features
xv_train = pd.concat([pd.DataFrame(xv_train.toarray()), train_sentiment_scores.reset_index(drop=True), train_pos_features.reset_index(drop=True)], axis=1)
xv_test = pd.concat([pd.DataFrame(xv_test.toarray()), test_sentiment_scores.reset_index(drop=True), test_pos_features.reset_index(drop=True)], axis=1)

# Step 5: Train and Evaluate Models
def evaluate_model(model, model_name):
    model.fit(xv_train, y_train)
    y_pred = model.predict(xv_test)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Logistic Regression
evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

# Decision Tree
evaluate_model(DecisionTreeClassifier(), "Decision Tree")

# Random Forest
evaluate_model(RandomForestClassifier(random_state=0), "Random Forest")

# Naive Bayes
evaluate_model(MultinomialNB(), "Naive Bayes")
