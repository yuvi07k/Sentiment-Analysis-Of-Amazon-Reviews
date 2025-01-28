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


## File Descriptions
demo.py: A concise script for running the sentiment analysis on smaller datasets. It demonstrates preprocessing, model training, and evaluation.

unittests.py: Contains unit tests for core functions, ensuring correctness in parsing, cleaning, sentiment labeling, and n-gram generation.

CS4120.Yuvraj.Kapoor.Final.Project.ipynb: The complete Jupyter notebook with detailed explanations, extended preprocessing techniques, and advanced generalizations for larger datasets.

Yuvraj Kapoor Report.pdf: A comprehensive project report detailing methodology, results, and discussions.

## Key Insights
Text Preprocessing: Lemmatization and stopword removal significantly improve model performance.

Resampling Techniques: Borderline SMOTE balances class distributions, enhancing accuracy.

Model Selection: Random Forests outperform other classifiers in handling feature-rich text data.

## Future Improvements
Fine-tune models for better performance.

Experiment with deep learning models like LSTMs and BERT.

Incorporate additional features such as reviewer profiles and review length.

