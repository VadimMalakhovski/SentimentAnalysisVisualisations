"""
NAME
    vader

DESCRIPTION
    # Program Name: vader.py
    # Purpose: evaluate vader
    # Example Of: Functions of vader's evaluation
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocess import X_test_vader, y_test
import sklearn
from sklearn.metrics import accuracy_score


# pre process for vader - remove links, other langs, @usernames
analyzer = SentimentIntensityAnalyzer()

vaders_predictions = []
# translating vader's scores into the ones used by this project 0=negative 2=neutral and 4=positive
for tweet in X_test_vader:
    tmp = analyzer.polarity_scores(tweet)
    if float(tmp['compound']) <= -0.344:
        vaders_predictions.append(0)
    elif float(tmp['compound'])  > 0.302:
        vaders_predictions.append(4)
    else:
        vaders_predictions.append(2)

# vader accuracy calculation
y_test = list(y_test)
# get accuracy
vader_accuracy = sklearn.metrics.accuracy_score(y_test, vaders_predictions)
vader_score = str(round(float(vader_accuracy), 3))

