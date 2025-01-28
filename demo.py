#!/usr/bin/env python
# coding: utf-8

#pip install numpy
#pip install pandas
#pip install nltk
#nltk.download('wordnet')
#nltk.download('stopwords') 
#pip install scikit-learn
#pip install imbalanced-learn

# In[1]:


import gzip
import json
import pandas as pd 
import numpy as np 
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn import preprocessing 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE


# In[2]:


def parse(path):
    """
    Generator function that reads a gzip compressed file containing JSON entries.
    
    Args:
    path (str): The file path to the gzip-compressed file.

    Yields:
    dict: Each line in the file is a JSON object, converted to a Python dictionary.
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    """
    Constructs a pandas DataFrame from a gzip compressed file containing JSON objects on each line.
    
    Args:
    path (str): The file path to the gzip-compressed file.

    Returns:
    DataFrame: A pandas DataFrame where each row corresponds to one JSON object.
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# In[3]:


def sentiment_review(row):  
    """
    Determines the sentiment of a review based on the 'overall' rating.

    Args:
    row (dict or pd.Series): A dictionary or a pandas Series object representing a single row of a DataFrame.

    Returns:
    str: The sentiment of the review ('Neutral', 'Negative', 'Positive') or -1 if the rating is outside the expected range.
    """
    if row['overall'] == 3.0:
        val = 'Neutral'
    elif row['overall'] == 1.0 or row['overall'] == 2.0:
        val = 'Negative'
    elif row['overall'] == 4.0 or row['overall'] == 5.0:
        val = 'Positive'
    else:
        val = -1
    return val

def clean_text(text):
    """Remove stopwords and perform lemmatization on text."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Remove punctuation except for exclamation points and question marks
    retain_punctuation = '!?'
    pattern = '[%s]' % re.escape(string.punctuation.replace('!', '').replace('?', ''))
    text = re.sub(pattern, '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stop_words= ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "to", "from",
             "in", "out", "on", "again","further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
             "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", 
             "too", "very", "can", "will", "just", "should", "now"]
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()])
    return cleaned_text

def prepare_data(df):
    """Preprocess and prepare data for modeling."""
    df['cleaned_text'] = df['reviewText'].apply(clean_text)
    df['sentiment'] = df.apply(sentiment_review, axis=1)
    return df


# In[4]:


def train_models(df):
    """Train and evaluate models on the dataset."""
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Handling class imbalance with SMOTE
    smote = BorderlineSMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_test)
    print("Decision Tree Confusion Matrix:")
    print(confusion_matrix(y_test, dt_pred))
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, dt_pred))

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred))


# In[5]:


if __name__ == '__main__':
    #Uncomment all to run and choose any one path 
    
    path = 'Appliances_5.json.gz' #less time(1 minute)
    #path = 'mydata.json.gz' #little more time(2.5 minutes)
    
    df = getDF(path)
    df = prepare_data(df)
    train_models(df)


# In[6]:


#Here, the accuracy for the appliances_5 file is extremely high since it only has 2500 rows, but the dataset mydata
#is the one I created using the different reviews from the amazon review data site. I combined reviews from 
#different datasets such as luxury, software, books etc. The accuracy for that is 77%.
#My implementation in the jupyter file is better since this is a demo file which I created to be concise and take 
#less time to run. The purpose was to see that the code is being run. The jupyter .ipynb file has a lot more
#techniques and generalization. The models also do not have any fine-tuning owing to the time constraints as it 
#would take a long time to run 

