#!/usr/bin/env python
# coding: utf-8

# In[43]:


import unittest
from unittest.mock import patch
import gzip
import json
import pandas as pd
from io import BytesIO
import re
import string


# In[44]:


def parse(path):
    """ Generator function that reads a gzip compressed file containing JSON entries. """
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    """ Constructs a pandas DataFrame from a gzip compressed file containing JSON objects on each line. """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

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

def clean(text):
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
    
    return text

STOPWORDS = set(["the", "is", "at", "which", "and", "on"])

def generate_ngrams(text, n_gram=1):
    """Generate n-grams from text."""
    token = [token for token in text.lower().split(" ") if token != "" and token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


# In[45]:


class TestYourModule(unittest.TestCase):
    
    def setUp(self):
        self.json_data = b'{"name": "John", "age": 30}\n{"name": "Jane", "age": 25}\n'
        self.expected_dicts = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25}
        ]

    @patch('gzip.open')
    def test_parse(self, mock_gzip_open):
        mock_gzip_open.return_value = BytesIO(self.json_data)
        result = list(parse('dummy_path.gz'))
        self.assertEqual(result, self.expected_dicts)
        
    @patch('__main__.parse', return_value=iter([])) 
    def test_getDF(self, mock_parse):
        mock_parse.return_value = iter(self.expected_dicts)
        result_df = getDF('dummy_path.gz')
        expected_df = pd.DataFrame.from_records(self.expected_dicts)
        pd.testing.assert_frame_equal(result_df, expected_df)
        
    def test_neutral_sentiment(self):
        row = {'overall': 3.0}
        self.assertEqual(sentiment_review(row), 'Neutral')

    def test_negative_sentiment_low(self):
        row = {'overall': 1.0}
        self.assertEqual(sentiment_review(row), 'Negative')

    def test_negative_sentiment_high(self):
        row = {'overall': 2.0}
        self.assertEqual(sentiment_review(row), 'Negative')

    def test_positive_sentiment_low(self):
        row = {'overall': 4.0}
        self.assertEqual(sentiment_review(row), 'Positive')

    def test_positive_sentiment_high(self):
        row = {'overall': 5.0}
        self.assertEqual(sentiment_review(row), 'Positive')

    def test_invalid_rating_lower_bound(self):
        row = {'overall': 0.0}
        self.assertEqual(sentiment_review(row), -1)

    def test_invalid_rating_upper_bound(self):
        row = {'overall': 6.0}
        self.assertEqual(sentiment_review(row), -1)

    def test_invalid_rating_non_numeric(self):
        row = {'overall': 'five'}
        self.assertEqual(sentiment_review(row), -1)

    def test_missing_overall_key(self):
        row = {'rating': 4.0}
        with self.assertRaises(KeyError):
            sentiment_review(row)

    def test_input_as_pandas_series(self):
        row = pd.Series({'overall': 4.0})
        self.assertEqual(sentiment_review(row), 'Positive')
        
    def test_lower_case(self):
        self.assertEqual(clean("HELLO WORLD"), "hello world")

    def test_remove_url(self):
        self.assertEqual(clean("Check this out: https://example.com"), "check this out ")

    def test_remove_html_tags(self):
        self.assertEqual(clean("Welcome to <b>Python</b>"), "welcome to python")

    def test_retain_important_punctuation(self):
        self.assertEqual(clean("Exciting!!! Right? Yes!"), "exciting!!! right? yes!")

    def test_remove_numbers(self):
        self.assertEqual(clean("There are 2 apples"), "there are  apples")
        
    def test_unigrams(self):
        self.assertEqual(generate_ngrams("hello world"), ["hello", "world"])
        
    def test_bigrams(self):
        self.assertEqual(generate_ngrams("hello beautiful world", n_gram=2), ["hello beautiful", "beautiful world"])
        
    def test_trigrams(self):
        self.assertEqual(generate_ngrams("hello very beautiful world", n_gram=3), ["hello very beautiful", "very beautiful world"])

    def test_stopwords_removal(self):
        self.assertEqual(generate_ngrams("the quick brown fox", n_gram=1), ["quick", "brown", "fox"])

    def test_empty_text(self):
        self.assertEqual(generate_ngrams("", n_gram=1), [])
        
    def test_no_stopwords_case(self):
        self.assertEqual(generate_ngrams("the is at", n_gram=1), [])


# In[46]:


# Running the tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestYourModule)
unittest.TextTestRunner().run(suite)


# In[ ]:




