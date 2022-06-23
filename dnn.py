"""
NAME
    dnn

DESCRIPTION
    # Program Name: dnn.py
    # Purpose: train and test dnn model
    # Example Of: Functions of ML compilation, fit and evaluation
"""
import sklearn
from preprocess import y_test, X_tests_dnn, my_dataset, last_20_percent_testing, first_80_percent_training
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K


# supress Keras Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


'''
IMPORTANT - doing some other preprocessing for DNN as this give better results
'''
# the maximum number of most frequent words to be used.
MAX_NB_WORDS = 280
# maximum number of words in each tweets.
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100

Y = pd.get_dummies(my_dataset.Sentiment_Score).values

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, split=' ')
tokenizer.fit_on_texts(my_dataset.Tweets.values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(my_dataset.Tweets.values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

####SEE IF THIS CAN BE REDUCED FROM CODE FROM TOP INTO 1 PREPROCESS
X_test_dnn = X[-last_20_percent_testing:] #last_20_percent_testing
# y values last shuffleed each time but can be used for accuracy for test of last 200
y_test_dnn  = Y[-last_20_percent_testing:]

X_train_dnn = X[:first_80_percent_training]
y_train_dnn = Y[:first_80_percent_training]


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', get_f1]) #categorical_crossentropy

epochs = 30
batch_size = 5

history_1 = model.fit(X_train_dnn, y_train_dnn,
                      verbose=0,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=0.2,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=5,  min_delta=0.00001)])



accr = model.evaluate(X_test_dnn,y_test_dnn)
###print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


'''
# TESTING WITH A NEW TWEET - Input your sentence in generate_sentiment.py
tweet = ["I had a great day yesterday although it was raining"]
seq = tokenizer.texts_to_sequences(tweet)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['negative','neutral','positive']
#print(pred, labels[np.argmax(pred)])
'''

dnn_predictions  = []

for tweet in X_tests_dnn:
    tweet = [tweet]
    seq = tokenizer.texts_to_sequences(tweet)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['negative', 'neutral', 'positive']
    if labels[np.argmax(pred)] == 'negative':
        dnn_predictions.append(0)
    elif labels[np.argmax(pred)] == 'positive':
        dnn_predictions.append(4)
    else:
        dnn_predictions.append(2)


# DNN accuracy calculation
y_test = list(y_test)
dnn_accuracy = sklearn.metrics.accuracy_score(y_test, dnn_predictions)
dnn_score = str(round(float(dnn_accuracy), 3))