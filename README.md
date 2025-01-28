# Sentiment Analysis of Amazon Reviews

## Project Overview
This project performs sentiment analysis on Amazon product reviews using machine learning techniques. It predicts customer satisfaction and sentiment (Positive, Neutral, Negative) for the reviews, enabling manufacturers and users to make data-driven decisions.

The project focuses on textual preprocessing, feature engineering, and classification models such as Logistic Regression, Decision Trees, Random Forests, and K-Nearest Neighbors. The dataset used is from Amazon's Luxury Beauty section, though it supports any 5-core Amazon review dataset.

## Features
- **Sentiment Prediction**: Classifies reviews as Positive, Neutral, or Negative.
- **Text Preprocessing**: Cleaning, tokenization, lemmatization, and TF-IDF vectorization.
- **Class Imbalance Handling**: Utilizes Borderline SMOTE for balanced model training.
- **Machine Learning Models**: Implements various classifiers to achieve high accuracy.
- **Evaluation Metrics**: Outputs confusion matrix, classification report, and other metrics.

## Dataset
The dataset includes Amazon reviews with metadata such as:
- Ratings (1–5 stars)
- Review text and summary
- Verified purchase status

For more information on the dataset, visit [Amazon Product Data](https://nijianmo.github.io/amazon/).

## Requirements
Install the required libraries using:

pip install numpy pandas nltk scikit-learn imbalanced-learn
nltk.download('wordnet')
nltk.download('stopwords')
