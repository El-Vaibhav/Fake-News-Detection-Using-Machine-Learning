import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import string
import numpy as np

nltk.download('vader_lexicon')  # Download Vader lexicon for sentiment analysis

# Load Data
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# Adding a 'class' column: fake_news -> 0, real_news -> 1
fake_news["class"] = 0
real_news["class"] = 1

# Manual testing datasets (not used in training)
fake_news_manual_testing = fake_news.tail(10)
real_news_manual_testing = real_news.tail(10)

# Dropping the last 100 rows from the main datasets for training
real_news = real_news.iloc[:-100]
fake_news = fake_news.iloc[:-100]

# Combine datasets
combined_data = pd.concat([fake_news, real_news], axis=0)

# Remove unnecessary columns
final_data = combined_data.drop(["title", "subject", "date"], axis=1)

# Shuffle the data
final_data = final_data.sample(frac=1).reset_index(drop=True)

# Combine manual testing datasets
combined_data_manual_testing = pd.concat([fake_news_manual_testing, real_news_manual_testing], axis=0)
final_manual_testing = combined_data_manual_testing.drop(["title", "subject", "date"], axis=1)
final_manual_testing = final_manual_testing.sample(frac=1).reset_index(drop=True)

# Preprocess Data and Split into Train/Test
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
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature Extraction with TF-IDF and Sentiment Analysis
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

# Concatenate sentiment scores with TF-IDF features
xv_train = np.concatenate([xv_train.toarray(), train_sentiment_scores.values.reshape(-1, 4)], axis=1)
xv_test = np.concatenate([xv_test.toarray(), test_sentiment_scores.values.reshape(-1, 4)], axis=1)

# Train models
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)

# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

# Random Forest
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

# Function to output label based on prediction
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Function for manual testing
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = vectorizer.transform(new_def_test['text'])
    test_sentiment_scores = new_def_test['text'].apply(calculate_sentiment_scores)
    new_x_test = np.concatenate([new_x_test.toarray(), test_sentiment_scores.values.reshape(-1, 4)], axis=1)
    
    pred_LR = LR.predict(new_x_test)
    pred_DT = DT.predict(new_x_test)
    pred_RF = RF.predict(new_x_test)
    
    return {
        "LR Prediction": output_label(pred_LR[0]),
        "DT Prediction": output_label(pred_DT[0]),
        "RFC Prediction": output_label(pred_RF[0]),
    }

def test_all_manual_testing(data):
    results = []
    for index, row in data.iterrows():
        news = row['text']
        true_label = row['class']
        predictions = manual_testing(news)
        results.append({
            'News': news,
            'True Label': output_label(true_label),
            **predictions
        })
    return pd.DataFrame(results)

# Test all entries in final_manual_testing
results_df = test_all_manual_testing(final_manual_testing)

# Display the results
print(results_df)

# Optionally, save the results to a CSV file
results_df.to_csv('manual_testing_results.csv', index=False)
