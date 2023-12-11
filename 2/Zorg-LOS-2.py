# author: Sjoerd van Dorp, 1032109, 11/12/2023, version 1.0
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
from sklearn.metrics import accuracy_score
from lazypredict.Supervised import LazyClassifier
import xgboost
import warnings

# Set the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'Zorg-LOS.csv' file
file_path = "C:\\Users\\sjoer\\OneDrive\\Documents\\HR\\Jaar 3\\OP2\\Predictive Analytics\\Tussentijdse opdrachten\\1\\Zorg-LOS.csv"
df = pd.read_csv(file_path)

#Opdracht 2
#---------------------------------------------------------

#Opsplitsen Train en Test Datasets
traindf = df[df['Stay']!=-1]
testdf = df[df['Stay']==-1]

# Opsplitsen train data
X1 = traindf.drop('Stay', axis=1)
y1 = traindf['Stay']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size =0.30, random_state =100)
print("\nTrain data:")
print(X_train.head())

# Outlier detectie
def detect_outliers(df, n, features):
    outlier_indices = []

    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

# outlier detectie voor getallen
outliers_to_drop = detect_outliers(df, 2, ["Hospital_code", 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital',
                                           'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness',
                                           'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay'])
# Drop outliers
df.drop(df.loc[outliers_to_drop].index, inplace=True)
print("\nOutliers verwijderen...")

# Normaal verdeling plot
selected_column = 'Admission_Deposit'
plt.figure(figsize=(10, 6))
sns.histplot(df[selected_column], bins=30, kde=True, color='blue', stat='density')

mean_value, std_dev_value = df[selected_column].mean(), df[selected_column].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_value, std_dev_value)
plt.plot(x, p, 'k', linewidth=2)

plt.title(f'Normaal Verdeling van {selected_column}')
plt.xlabel('Waarden')
plt.ylabel('Frequentie')
plt.show()

# heatmap aanmaken
plt.figure(figsize=(13,10))
sns.heatmap(traindf.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

# Kolommen voor de boxplot
columns_for_boxplot = ['Available Extra Rooms in Hospital', 'Visitors with Patient']

# boxplot maken voor geselecteerde kolommen
sns.boxplot(data=df[columns_for_boxplot])
plt.title('Boxplot')
plt.show()

# Modellen trainen
print("\nDe accuracy van de modellen worden geprint: ")

# Gradient Boosting Model
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=100)
gb_model = gb_classifier.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(gb_predictions, y_test)
print("\nGradient Boosting Classifier Accuracy:", gb_accuracy * 100)

# Naive Bayes Model
target = y_train.values
features = X_train.values
classifier_nb = GaussianNB()
model_nb = classifier_nb.fit(features, target)
prediction_nb = model_nb.predict(X_test)
acc_score_nb = accuracy_score(prediction_nb,y_test)
print("\nNaive Bayes Acurracy:", acc_score_nb*100)

# XGBoost Model
classifier_xgb = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1,
n_estimators=800,objective='multi:softmax', reg_alpha=0.5,
reg_lambda=1.5, booster='gbtree', n_jobs=4, min_child_weight=2, base_score= 0.75)
model_xgb = classifier_xgb.fit(X_train, y_train)
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = accuracy_score(prediction_xgb,y_test)
print("\nXGboost Accuracy:", acc_score_xgb*100)

# RandomForest Model
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=100)
model_rf = classifier_rf.fit(X_train, y_train)
prediction_rf = model_rf.predict(X_test)
acc_score_rf = accuracy_score(prediction_rf, y_test)
print("\nRandom Forest Accuracy:", acc_score_rf * 100)

print("\nEr is te zien dat de XGboost model de hoogste accuracy heeft. Laten we voor dit model een confusion matrix en AOC maken: ")

# Confusion matrix en AOC
y_test_bin = label_binarize(y_test, classes=model_xgb.classes_)
y_prob_xgb_ovr = model_xgb.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(model_xgb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_xgb_ovr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot AOC curves voor elke class
plt.figure(figsize=(10, 8))
for i in range(len(model_xgb.classes_)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {model_xgb.classes_[i]} (AUC = {roc_auc[i]:.2f})')

# Plot AOC curve voor random keuzes
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AOC Curve for XGBoost (One-vs-Rest)')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix voor XGBoost Model
xgb_conf_matrix = confusion_matrix(y_test, prediction_xgb)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=model_xgb.classes_, yticklabels=model_xgb.classes_)
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
