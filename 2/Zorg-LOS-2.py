# author: Sjoerd van Dorp, 1032109, 29/11/2023, version 1.0
# Tussentijdse Opdracht 2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # to plot charts
from datetime import date
from collections import Counter
import os
# Modeling Libraries
from sklearn.preprocessing import QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split
from lazypredict.Supervised import LazyRegressor, LazyClassifier
import warnings

# Set the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'Zorg-LOS.csv' file
file_path = "C:\\Users\\sjoer\\OneDrive\\Documents\\HR\\Jaar 3\\OP2\\Predictive Analytics\\Tussentijdse opdrachten\\1\\Zorg-LOS.csv"
df = pd.read_csv(file_path)

# Splitting train data for Naive Bayes and XGBoost
X1 = df.drop('Stay', axis =1)
y1 = df['Stay']

X_train, X_test, y_train, y_test = train_test_split(X1, y1,
test_size =0.10, random_state =100)

# Modellen testen

target = y_train.values
features = X_train.values
classifier_nb = GaussianNB()
model_nb = classifier_nb.fit(features, target)
prediction_nb = model_nb.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score_nb = accuracy_score(prediction_nb,y_test)
print("\nGaussianNB Acurracy:", acc_score_nb*100)

# XGBoost Model
import xgboost
classifier_xgb = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1,
n_estimators=800,objective='multi:softmax', reg_alpha=0.5,
reg_lambda=1.5, booster='gbtree', n_jobs=4, min_child_weight=2, base_score= 0.75)
model_xgb = classifier_xgb.fit(X_train, y_train)
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = accuracy_score(prediction_xgb,y_test)
print("\nXGboost Accuracy:", acc_score_xgb*100)

# LazyClassifier Instance and fitting data
cls= LazyClassifier(ignore_warnings=False,
custom_metric=None)
models, predictions = cls.fit(X_train, X_test, y_train, y_test)
print(models.to_string())
