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

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/97830a2f-660c-432a-9d5d-fec761cceedf)


2) Final Output with testing Dataset

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/71f39af6-10c7-4013-85dd-7df64b2b9020)

# TFIDF 

1) Classification Report

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/20ee71cf-8e45-4839-8766-02cfe9270dc3)


2) Final Output with testing Dataset

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/9ce980fc-56df-4e39-b317-af894da8aead)

# Word Count

1) Classification Report
   
![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/d75733b3-2250-4b01-a6e3-4ff213d58138)


3) Final Output with testing Dataset
   
![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/0dc6bcd6-de9a-45e2-a6e2-c6a02876cf7a)

# Sentiment Analysis

1) Classification Report

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/36adfa5d-c38b-43ac-94f4-2e1ff5965cbe)

2) Final Output with testing Dataset

![image](https://github.com/El-Vaibhav/Fake-News-Detection-Using-Machine-Learning/assets/135622906/9240f3c4-0cbc-4252-a54a-bdb4ddffb621)


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
 
