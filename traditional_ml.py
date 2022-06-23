"""
NAME
    traditional_ml

DESCRIPTION
    # Program Name: traditional_ml.py
    # Purpose: train and test traditional ml models
    # Example Of: Functions of MLs compilation,  fit and evaluation
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from preprocess import X_train, y_train, X_test, y_test


# Training Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)
y_predict_nb = naive_bayes_model.predict(X_test)
nb_score = str(round(accuracy_score(y_test, y_predict_nb), 3))

# Training Logistic Regression model
logistic_reg_model = LogisticRegression(solver='lbfgs', max_iter=10000)
logistic_reg_model.fit(X_train, y_train)
y_predict_lr = logistic_reg_model.predict(X_test)
lr_score = str(round(accuracy_score(y_test, y_predict_lr), 3))


# Training SVC model
svc=SVC(probability=True, kernel="linear", class_weight="balanced")
cv=cross_val_score(svc, X_train, y_train.ravel(), cv=10)
model = svc.fit(X_train, y_train.ravel())
score = svc.score(X_test, y_test)
svc_score = str(round(score, 3))
