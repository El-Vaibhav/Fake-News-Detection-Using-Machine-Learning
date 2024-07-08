# Fake News Detection Using Machine Learning

This project focuses on the detection of fake news using machine learning algorithms. Four different classifiers (Logistic Regression, Decision Tree, Random Forest, and Naive Bayes) were evaluated across four distinct feature sets: Word Count, N-Gram Count, TF-IDF, and Sentiment Analysis. The goal is to determine the most effective feature set and classifier for fake news detection.

# Table of Contents
1) Overview
2) Project Structure
3) Datasets
4) Feature Extraction
5) Model Training and Evaluation
6) Results
7) Conclusion
8) Usage
9) Contributing
10) License

# Datasets
The datasets used in this project are:

I have used the ISOT Fake News , Real News Dataset for this project there are two csv files for the dataset
1) True.csv
2) Fake.csv

https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

Both datasets contain news snippets labeled as "Fake News" or "Not A Fake News".

# Feature Extraction
Four different feature extraction methods were used:

1) Word Count
2) N-Gram Count
3) TF-IDF
4) Sentiment Analysis
5) Model Training and Evaluation

Four classifiers were used to train the models and evaluate their performance:

1) Logistic Regression (LR)
2) Decision Tree (DT)
3) Random Forest (RF)
4) Naive Bayes (NB)
5) Each classifier was evaluated using each feature set to determine the accuracy of fake news detection.

# ScreenShots

1) N Gram Count

   i)Classification Report
    ![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/1cbeabe0-4aca-4d65-a6f2-7ef59efccece)



# Results 

The performance of each classifier with different feature sets is summarized below:

The testing was made with the last 20 articles of each dataset ( Fake and Real)

Word Count Feature

1) Logistic Regression (LR): 50% Accuracy
2) Decision Tree (DT): 50% Accuracy
3) Random Forest (RF): 45% Accuracy
4) Naive Bayes (NB): 50% Accuracy
   
N-Gram Count Feature

1) Logistic Regression (LR): 100% Accuracy
2) Decision Tree (DT): 100% Accuracy
3) Random Forest (RF): 100% Accuracy
4) Naive Bayes (NB): 90% Accuracy
   
TF-IDF Feature

1) Logistic Regression (LR): 100% Accuracy
2) Decision Tree (DT): 100% Accuracy
3) Random Forest (RF): 100% Accuracy
4) Naive Bayes (NB): 90% Accuracy
   
Sentiment Analysis Feature

1) Logistic Regression (LR): 100% Accuracy
2) Decision Tree (DT): 100% Accuracy
3) Random Forest (RF): 100% Accuracy

# Conclusion
The analysis shows that the N-Gram Count, TF-IDF, and Sentiment Analysis features are highly effective for fake news detection, achieving perfect or near-perfect accuracy. The Word Count feature, however, is less reliable and should not be used for this purpose.

# Usage
Clone the repository:
https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning.git

# Navigate to the project directory:
cd Fake-News-Detection-Using-Machine-Learning

# Install the required packages:
pip install -r requirements.txt

Run the data preprocessing, feature extraction, and model training scripts as needed.

# Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

# License
This project is licensed under the MIT License.
 
