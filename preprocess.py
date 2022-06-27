"""
NAME
    preprocess

DESCRIPTION
    # Program Name: preprocess.py
    # Purpose: cleaning, formatting, vectorizing, stemming, lemmatizing, randomizing and splitting tweets
    # Example Of: Functions of cleaning and formatting tweets
"""
import re
import pandas as pd
import numpy as np
import string
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = set(stopwords.words('english'))

def clean_and_format_tweets(tweet, is_vader=False):
    tweet.lower()
    # Remove hash  # and user @ references
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # remove subject elon musk
    tweet = tweet.replace('elon', '')
    tweet = tweet.replace('musk', '')
    # making sure  vader gets a clean tweet but without other preprocessing like stemming
    if is_vader ==True:
        return tweet
    else:
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in stop_words]
        # chose either stemming or lemmatizing, do not use both
        # stemming
        porter_streamer = PorterStemmer()
        stemmed_words = [porter_streamer.stem(w) for w in filtered_words]
        # lemmatizing
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

        return " ".join(filtered_words)


def vectorization(x_data):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(x_data)
    return vector

# Load Dataset
my_dataset = pd.read_csv("tweets_data/tweets_annotated_compiled.csv", encoding='latin-1')

# Shuffle with pandas with their version of seed
# SHUFFLING IN SUCH A WAY TO MAKE SURE THAT MY ANNOTATED DATA IS IN TRAINING DS AS MY DS IS THE FIRST 212 ROWS SO THAT
# THE MODELS WILL BE TESTED AGAINST DIFFERENT DOMAINS
my_dataset = pd.concat([my_dataset[:213],my_dataset[1:].sample(frac=1, random_state=0)]).reset_index(drop=True)

# Preprocess data
# making sure vader gets a clean tweet but without other preprocessing like stemming
my_dataset_vader = my_dataset.copy()
my_dataset_vader.Tweets = my_dataset_vader['Tweets'].apply(clean_and_format_tweets, is_vader=True)
my_dataset.Tweets = my_dataset['Tweets'].apply(clean_and_format_tweets, is_vader=False)


# Split 80-20 and vectorise
# get n of rows
n_rows = my_dataset.shape[0]

# first shuffled 80%
first_80_percent_training = int(n_rows * 0.8)

# last shuffled 20%
last_20_percent_testing = int(n_rows * 0.2)

X_tests_dnn = np.array(my_dataset.iloc[:, 0].tail(last_20_percent_testing)).ravel()
X_test_vader = np.array(my_dataset_vader.iloc[:, 0].tail(last_20_percent_testing)).ravel()
y_test = np.array(my_dataset.iloc[:, 1].tail(last_20_percent_testing)).ravel()

# SVM test split for comparison with VADER
tf_vector_last_20_percent = vectorization(np.array(my_dataset.iloc[:, 0]).ravel())
X_test = tf_vector_last_20_percent.transform(np.array(my_dataset.iloc[:, 0].tail(last_20_percent_testing)).ravel())

# vectorisation for training
tf_vector = vectorization(np.array(my_dataset.iloc[:, 0]).ravel())
X_train = tf_vector.transform(np.array(my_dataset.iloc[:, 0].head(first_80_percent_training)).ravel())
y_train = np.array(my_dataset.iloc[:, 1].head(first_80_percent_training)).ravel()


