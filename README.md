# Fake News Detection Using Machine Learning

This project focuses on the detection of fake news using machine learning algorithms. Four different classifiers (Logistic Regression, Decision Tree, Random Forest, and Naive Bayes) were evaluated across four distinct feature sets: Word Count, N-Gram Count, TF-IDF, and Sentiment Analysis. The goal is to determine the most effective feature set and classifier for fake news detection.

# Table of Contents
1) Overview
2) Datasets
3) Feature Extraction
4) Screenshots
5) Results
6) Conclusion
7) Usage
8) Contributing
9) License

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

# N Gram Count

1) Classification Report

![image](https://github.com/user-attachments/assets/af441be6-f381-45d0-a7c9-f1b03ce23bdc)
![image](https://github.com/user-attachments/assets/a64fc1c5-7d46-4306-896c-9b90ef833e57)
![image](https://github.com/user-attachments/assets/2e66cbec-3a08-4929-84dc-06f85d9f2213)


2) Final Output with testing Dataset

![image](https://github.com/user-attachments/assets/7684e32e-b536-4906-bfb8-d658e7fab7f8)


# TFIDF 

1) Classification Report

![image](https://github.com/user-attachments/assets/b894ce77-4674-4434-afb3-5ac8f45556a7)
![image](https://github.com/user-attachments/assets/d47aaa87-cae5-4895-924b-a72424ddfbbc)


2) Final Output with testing Dataset

![image](https://github.com/user-attachments/assets/3eecae21-da21-4a04-81cf-1a0d4d505279)


# Word Count

1) Classification Report
   
![image](https://github.com/user-attachments/assets/e70d413d-c40a-4a22-9efe-729c4dd438ca)
![image](https://github.com/user-attachments/assets/953cb370-68d4-450f-b180-c8d78031e9f1)
![image](https://github.com/user-attachments/assets/f5562c86-2e0c-4606-a4b1-baf656e2f277)


3) Final Output with testing Dataset
   
![image](https://github.com/user-attachments/assets/01547b0a-ad1c-4117-a830-a4ccb226e7f4)


# Sentiment Analysis

1) Classification Report

![image](https://github.com/user-attachments/assets/fb0ccd24-98d0-4148-a7da-ad6c71b4bef3)
![image](https://github.com/user-attachments/assets/f29a6740-38b7-4c5d-a26d-52bea4a0a161)


2) Final Output with testing Dataset

![image](https://github.com/user-attachments/assets/b4440251-3006-4f01-b278-2e23d222b86a)


# Results 

# Objective:
The primary aim of this project was to evaluate different feature extraction techniques and classifiers for the task of fake news detection. The goal was to determine which combination of feature and classifier works best in terms of accuracy and performance on this specific task.

# Feature Extraction Techniques:
1.	Word Count: The number of words in each news article was used as a feature.
2.	N-Gram Count: This feature considered the frequency of sequences of 'n' words (bigrams, trigrams, etc.).
3.	TF-IDF (Term Frequency-Inverse Document Frequency): This feature measures how important a word is to a document in a collection.
4.	Sentiment Analysis: This feature captures the overall sentiment (positive or negative) of the news article.

# Classifiers Used:
1.	Logistic Regression (LR)
2.	Decision Tree (DT)
3.	Random Forest (RF)
4.	Naive Bayes (NB)

# Results Analysis:
•	Word Count Feature: Decision Tree and Random Forest classifiers performed exceptionally well, achieving 100% accuracy. Logistic Regression also performed strongly with nearly perfect
accuracy. However, Naive Bayes struggled with this feature, achieving only 50% accuracy.

•	N-Gram Count Feature: Logistic Regression was the standout performer, achieving perfect accuracy, while other classifiers, including Decision Tree, Random Forest, and Naive Bayes, reached only 61% accuracy.

•	TF-IDF Feature: Decision Tree and Random Forest again excelled with 100% accuracy, but Logistic Regression and Naive Bayes lagged behind with only 61% accuracy.

•	Sentiment Analysis Feature: Decision Tree and Random Forest were again top performers with 99.2% accuracy. Logistic Regression achieved 78.8% accuracy, while Naive Bayes remained at 50%.
Conclusion:

# Best Features: 
The Word Count and TF-IDF features, in combination with Decision Tree and Random Forest classifiers, provided the best results for fake news detection, with 100% accuracy in multiple tests.

# Best Classifiers: 
Decision Tree and Random Forest consistently outperformed other classifiers across different feature sets. Logistic Regression performed well with the N-Gram Count feature but showed lower accuracy with other features. Naive Bayes underperformed across most feature sets, achieving the highest accuracy (61%) only with N-Gram Count and TF-IDF features.

# Conclusion
Overall, Decision Tree and Random Forest classifiers with Word Count or TF-IDF features appear to be the most effective combination for this fake news detection task.

# Usage
Clone the repository:
https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning.git

# Navigate to the project directory:
cd Fake-News-Detection-Using-Machine-Learning
Run the data preprocessing, feature extraction, and model training scripts as needed.

# Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

# License
This project is licensed under the MIT License.
 
