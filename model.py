# Step 1: Import all the important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes

# Step 2: Load Data into two variables: one is fake and one is real
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

# Merge fake and real news datasets along the row axis=0
combined_data = pd.concat([fake_news, real_news], axis=0)

# Remove unnecessary columns like 'title', 'subject', 'date' and store in final_data
final_data = combined_data.drop(["title", "subject", "date"], axis=1)

# Randomly shuffle the rows of data
final_data = final_data.sample(frac=1).reset_index(drop=True)

combined_data_manual_testing = pd.concat([fake_news_manual_testing, real_news_manual_testing], axis=0)
final_manual_testing = combined_data_manual_testing.drop(["title", "subject", "date"], axis=1)
final_manual_testing = final_manual_testing.sample(frac=1).reset_index(drop=True)

# ----------------------------------
# DATA PREPROCESSING
# ------------------------------------

# Function to clean text data
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply wordopt function to clean 'text' column
final_data['text'] = final_data['text'].apply(wordopt)

# x is independent var and y is dependent
x = final_data['text']
y = final_data['class']

# training the model on 30% of dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# ----------------------------------------
# Feature Extraction
# -----------------------------

# TFIDF Feature (Term freq inverse - inverse document freq)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# -----------------------------
# Classifier Training the dataset
# ----------------------

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
# LR.score(xv_test, y_test)
# print("Logistic Regression:")
# print(f" Accuracy: {accuracy_score(y_test, pred_lr)}")
# print(classification_report(y_test, pred_lr))

# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
# DT.score(xv_test, y_test)
# print("Decision Tree:")
# print(f" Accuracy: {accuracy_score(y_test, pred_dt)}")
# print(classification_report(y_test, pred_dt))

# Random Forest
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
RandomForestClassifier(random_state=0)
pred_rf = RF.predict(xv_test)
# RF.score(xv_test, y_test)
# print("Random Forest")
# print(f" Accuracy: {accuracy_score(y_test, pred_rf)}")
# print(classification_report(y_test, pred_rf))

# Naive Bayes
NB = MultinomialNB()
NB.fit(xv_train, y_train)
pred_nb = NB.predict(xv_test)
# print("Naive Bayes (Multinomial):")
# print(f" Accuracy: {accuracy_score(y_test, pred_nb)}")
# print(classification_report(y_test, pred_nb))

# ------------------------------------
# TESTING PHASE
# ----------------------------------------

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def test_all_manual_testing(data):
    results = []
    for index, row in data.iterrows():
        news = row['text']
        true_label = row['class']
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)
        pred_NB = NB.predict(new_xv_test)
        
        results.append({
            'News': news,
            'True Label': output_label(true_label),
            'LR Prediction': output_label(pred_LR[0]),
            'DT Prediction': output_label(pred_DT[0]),
            'RF Prediction': output_label(pred_RF[0]),
            'NB Prediction': output_label(pred_NB[0])
        })
    
    return pd.DataFrame(results)

# Test all entries in final_manual_testing
results_df = test_all_manual_testing(final_manual_testing)

# Display the results
print(results_df)

# Optionally, save the results to a CSV file
results_df.to_csv('manual_testing_results.csv', index=False)
