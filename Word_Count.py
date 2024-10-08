import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes

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

# Function to get word count
def get_word_count(text):
    words = text.split()
    return len(words)

# Apply word count to 'text' column
final_data['word_count'] = final_data['text'].apply(get_word_count)

# x is independent var and y is dependent
x = final_data[['word_count']]
y = final_data['class']

# training the model on 80% of dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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

# Random Forest
RF = RandomForestClassifier(random_state=0)
RF.fit(x_train, y_train)
pred_rf = RF.predict(x_test)
print("Random Forest")
print("Accuracy:", RF.score(x_test, y_test)*100 ,"%")
print(classification_report(y_test, pred_rf))

# Naive Bayes
NB = MultinomialNB()
NB.fit(x_train, y_train)
pred_nb = NB.predict(x_test)
print("Naive Bayes (Multinomial):")
print("Accuracy:", NB.score(x_test, y_test)*100 ,"%")
print(classification_report(y_test, pred_nb))

# ------------------------------------
# TESTING PHASE
# ----------------------------------------

def output_label(n,true_label):
    if n == true_label:
        return "Fake News"
    else:
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
        new_def_test['word_count'] = new_def_test["text"].apply(get_word_count)
        new_x_test = new_def_test[['word_count']]
        
        pred_LR = LR.predict(new_x_test)
        pred_DT = DT.predict(new_x_test)
        pred_RF = RF.predict(new_x_test)
        pred_NB = NB.predict(new_x_test)

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
            'True Label': "Fake News" if true_label == 0 else "Not Fake News",
            'LR Prediction': output_label(pred_LR, true_label),
            'DT Prediction': output_label(pred_DT, true_label),
            'RF Prediction': output_label(pred_RF, true_label),
            'NB Prediction': output_label(pred_NB, true_label)
        })
    
    return pd.DataFrame(results), correct_predictions

# Run the test on all manual testing data
results_df, correct_predictions = test_all_manual_testing(final_manual_testing)

# Display the results and number of correct predictions
print(results_df)
print("Correct Predictions:", correct_predictions)

# Optionally, save the results to a CSV file
results_df.to_csv('manual_testing_results_Word_Count.csv', index=False)