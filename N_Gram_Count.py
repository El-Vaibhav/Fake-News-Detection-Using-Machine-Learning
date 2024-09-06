import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB  

# Step 2: Load Data into two variables: one is fake and one is real
fake_news = pd.read_csv('Fake-News-Detection-Using-Machine-Learning\\Fake.csv')
real_news = pd.read_csv('Fake-News-Detection-Using-Machine-Learning\\True.csv')

# Adding a 'class' column: fake_news -> 0, real_news -> 1
fake_news["class"] = 0
real_news["class"] = 1

# Manual testing datasets (not used in training)
fake_news_manual_testing = fake_news.tail(1000)
real_news_manual_testing = real_news.tail(1000)

# Dropping the last 100 rows from the main datasets for training
real_news = real_news.iloc[:-1000]
fake_news = fake_news.iloc[:-1000]

# Merge fake and real news datasets along the row axis=0
combined_data = pd.concat([fake_news, real_news], axis=0)

# Remove unnecessary columns like 'title', 'subject', 'date' and store in final_data
final_data = combined_data.drop(["title", "subject", "date"], axis=1)

# Randomly shuffle the rows of data
final_data = final_data.sample(frac=1).reset_index(drop=True)

# Doing the same for test data
combined_data_manual_testing = pd.concat([fake_news_manual_testing, real_news_manual_testing], axis=0)
final_manual_testing = combined_data_manual_testing.drop(["title", "subject", "date"], axis=1)
final_manual_testing = final_manual_testing.sample(frac=1).reset_index(drop=True)

# ----------------------------------
# DATA PREPROCESSING
# ------------------------------------

# Function to clean text data
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply cleaning function to 'text' column.
final_data['text'] = final_data['text'].apply(clean_text)
final_manual_testing['text'] = final_manual_testing['text'].apply(clean_text)

# Function to get n-gram counts
def get_ngram_counts(text):
    vectorizer = CountVectorizer(ngram_range=(1,1))
    X = vectorizer.fit_transform(text)
    return X, vectorizer

# Extracting unigram (1-gram) counts
x_counts, vectorizer = get_ngram_counts(final_data['text'])

# x is independent variable and y is dependent variable
x = x_counts
y = final_data['class']

print(x)
# Splitting the dataset into training and testing (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# ----------------------------------------
# Classifier Training and Testing
# ----------------------

# Logistic Regression
LR = LogisticRegression()
LR.fit(x_train, y_train)
pred_lr = LR.predict(x_test)
print("Logistic Regression:")
print("Accuracy:", LR.score(x_test, y_test)*100, "%")
print(classification_report(y_test, pred_lr))

# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
pred_dt = DT.predict(x_test)
print("Decision Tree:")
print("Accuracy:", DT.score(x_test, y_test)*100, "%")
print(classification_report(y_test, pred_dt))

# Random Forest
RF = RandomForestClassifier(random_state=0)
RF.fit(x_train, y_train)
pred_rf = RF.predict(x_test)
print("Random Forest")
print("Accuracy:", RF.score(x_test, y_test)*100, "%")
print(classification_report(y_test, pred_rf))

# Naive Bayes
NB = MultinomialNB()
NB.fit(x_train, y_train)
pred_nb = NB.predict(x_test)
print("Naive Bayes (Multinomial):")
print("Accuracy:", NB.score(x_test, y_test)*100, "%")
print(classification_report(y_test, pred_nb))

# ------------------------------------
# TESTING PHASE
# ----------------------------------------

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def test_all_manual_testing(data):
    correct_predictions = {
        'LR': 0,
        'DT': 0,
        'RF': 0,
        'NB': 0
    }

    results = []
    for index, row in data.iterrows():
        news = row['text']
        true_label = row['class']
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(clean_text)
        new_x_test = vectorizer.transform(new_def_test['text'])
        
        pred_LR = LR.predict(new_x_test)[0]
        pred_DT = DT.predict(new_x_test)[0]
        pred_RF = RF.predict(new_x_test)[0]
        pred_NB = NB.predict(new_x_test)[0]

        if pred_LR == true_label:
            correct_predictions['LR'] += 1
        if pred_DT == true_label:
            correct_predictions['DT'] += 1
        if pred_RF == true_label:
            correct_predictions['RF'] += 1
        if pred_NB == true_label:
            correct_predictions['NB'] += 1
        
        results.append({
            'News': news,
            'True Label': output_label(true_label),
            'LR Prediction': output_label(pred_LR),
            'DT Prediction': output_label(pred_DT),
            'RF Prediction': output_label(pred_RF),
            'NB Prediction': output_label(pred_NB)
        })
    
    return pd.DataFrame(results), correct_predictions

# Run the test on all manual testing data
results_df, correct_predictions = test_all_manual_testing(final_manual_testing)

# Display the results and number of correct predictions
print(results_df)
print("Correct Predictions:", correct_predictions)

# Optionally, save the results to a CSV file
results_df.to_csv('manual_testing_results_ngram.csv', index=False)
