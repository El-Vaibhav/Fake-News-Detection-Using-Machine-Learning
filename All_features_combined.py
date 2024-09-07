import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import hstack
from textblob import TextBlob
import pickle

# Load Data into two variables: one is fake and one is real.
fake_news = pd.read_csv('Fake-News-Detection-Using-Machine-Learning\\Fake.csv')
real_news = pd.read_csv('Fake-News-Detection-Using-Machine-Learning\\True.csv')

# Adding a 'class' column: fake_news -> 0, real_news -> 1
fake_news["class"] = 0
real_news["class"] = 1

# Manual testing datasets (not used in training)
fake_news_manual_testing = fake_news.tail(1000)
real_news_manual_testing = real_news.tail(1000)

# Dropping the last 1000 rows from the main datasets for training
real_news = real_news.iloc[:-1000]
fake_news = fake_news.iloc[:-1000]

# Merge fake and real news datasets along the row axis=0
combined_data = pd.concat([fake_news, real_news], axis=0)
# Remove unnecessary columns like 'title', 'subject', 'date' and store in final_data
final_data = combined_data.drop(["title", "subject", "date"], axis=1)
# Randomly shuffle the rows of data
final_data = final_data.sample(frac=1).reset_index(drop=True)

# Doing the same for the manual testing dataset
combined_data_manual_testing = pd.concat([fake_news_manual_testing, real_news_manual_testing], axis=0)
final_manual_testing = combined_data_manual_testing.drop(["title", "subject", "date"], axis=1)
final_manual_testing = final_manual_testing.sample(frac=1).reset_index(drop=True)

# ----------------------------------
# DATA PREPROCESSING
# ------------------------------------

# Function to clean text data
def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply preprocess function to clean 'text' column
final_data['text'] = final_data['text'].apply(preprocess)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis to 'text' column
final_data['sentiment_score'] = final_data['text'].apply(get_sentiment)

# Calculate word count
final_data['word_count'] = final_data['text'].apply(lambda x: len(x.split()))

# Normalize sentiment score and word count
scaler_sentiment = MinMaxScaler()
final_data['sentiment_score_normalized'] = scaler_sentiment.fit_transform(final_data[['sentiment_score']])

scaler_word_count = StandardScaler()
final_data['word_count_normalized'] = scaler_word_count.fit_transform(final_data[['word_count']])

# Define features and target variable
x = final_data[['text', 'sentiment_score_normalized', 'word_count_normalized']]
y = final_data['class']

# Feature Extraction
# TF-IDF Feature (Term freq inverse document freq) with n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
xv = tfidf_vectorizer.fit_transform(x['text'])

# N-gram Count Feature
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_counts = ngram_vectorizer.fit_transform(x['text'])

# Combine TF-IDF, n-gram counts, sentiment score, and word count
additional_features = x[['sentiment_score_normalized', 'word_count_normalized']].values
combined_features = hstack([xv, ngram_counts, additional_features])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.2)

# ----------------------------------------
# Classifier Training the dataset
# ----------------------

# Logistic Regression
LR = LogisticRegression()
LR.fit(x_train, y_train)
pred_lr = LR.predict(x_test)
print("Logistic Regression:")
print("Accuracy:", LR.score(x_test, y_test)*100 ,"%")
print(classification_report(y_test, pred_lr))

# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
pred_dt = DT.predict(x_test)
print("Decision Tree:")
print("Accuracy:", DT.score(x_test, y_test)*100 ,"%")
print(classification_report(y_test, pred_dt))

# ------------------------------------
# TESTING PHASE
# ----------------------------------------

def output_label(n, true_label):
    return "Fake News" if n == true_label else "Not Fake News"

def test_all_manual_testing(data):

    correct_predictions = {
        'LR': 0,
        'DT': 0,
    }

    results = []
    for index, row in data.iterrows():
        news = row['text']
        true_label = row['class']
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(preprocess)
        
        # Process new data
        new_def_test['sentiment_score'] = new_def_test['text'].apply(get_sentiment)
        new_def_test['word_count'] = new_def_test['text'].apply(lambda x: len(x.split()))
        new_def_test['sentiment_score_normalized'] = scaler_sentiment.transform(new_def_test[['sentiment_score']])
        new_def_test['word_count_normalized'] = scaler_word_count.transform(new_def_test[['word_count']])
        
        new_x_text = tfidf_vectorizer.transform(new_def_test['text'])
        new_ngram_counts = ngram_vectorizer.transform(new_def_test['text'])
        new_additional_features = new_def_test[['sentiment_score_normalized', 'word_count_normalized']].values
        new_combined_features = hstack([new_x_text, new_ngram_counts, new_additional_features])
        
        pred_LR = LR.predict(new_combined_features)
        pred_DT = DT.predict(new_combined_features)
        
        if pred_LR == true_label:
            correct_predictions['LR'] += 1
        if pred_DT == true_label:
            correct_predictions['DT'] += 1
        
        
        results.append({
            'News': news,
            'True Label': "Fake News" if true_label == 0 else "Not Fake News",
            'LR Prediction': output_label(pred_LR, true_label),
            'DT Prediction': output_label(pred_DT, true_label),
            
        })
    
    return pd.DataFrame(results), correct_predictions

# Run the test on all manual testing data
results_df, correct_predictions = test_all_manual_testing(final_manual_testing)

# Display the results and number of correct predictions
print(results_df)
print("Correct Predictions:", correct_predictions)

# Optionally, save the results to a CSV file
results_df.to_csv('manual_testing_results_TFIDF_Ngram.csv', index=False)
