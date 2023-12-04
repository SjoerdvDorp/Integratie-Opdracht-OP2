# author: Sjoerd van Dorp, 1032109, 29/11/2023, version 1.0
# Tussentijdse Opdracht 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'train.csv' file
file_path = "C:\\Users\\sjoer\\OneDrive\\Documents\\HR\\Jaar 3\\OP2\\Predictive Analytics\\Tussentijdse opdrachten\\1\\Zorg-LOS.csv"
df = pd.read_csv(file_path)

X = df['Ward_Type']
Y = df['Stay']

# Train/test split
X_train, Y_train = train_test_split(df, test_size=0.3, random_state=100)

# Aantal ontbrekende waarden in de data set
print("\nOntbrekende waarden in Data Set:")
print(df.isnull().sum())

unique_stay_values_y = df['Ward_Type'].unique()
unique_stay_values_x = df['Stay'].unique()

print("\nWaarden uit de Y set:")
print(unique_stay_values_y)
print("\nWaarden uit de X set:")
print(unique_stay_values_x)

# Modellen testen

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, Y_train)
lr_pred = lr_model.predict(X_test)

# Support Vector Machine
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, Y_train)
svm_pred = svm_model.predict(X_test)

# Evaluatie en visualisatie

# Confusion Matrix, AUC, Nauwkeurigheid en Classificatie Rapport voor Random Forest
rf_conf_matrix = confusion_matrix(Y_test, rf_pred)
rf_accuracy = accuracy_score(Y_test, rf_pred)
rf_roc_auc = roc_auc_score(Y_test, rf_pred)

# Confusion Matrix, AUC, Nauwkeurigheid en Classificatie Rapport voor Logistic Regression
lr_conf_matrix = confusion_matrix(Y_test, lr_pred)
lr_accuracy = accuracy_score(Y_test, lr_pred)
lr_roc_auc = roc_auc_score(Y_test, lr_pred)

# Confusion Matrix, AUC, Nauwkeurigheid en Classificatie Rapport voor SVM
svm_conf_matrix = confusion_matrix(Y_test, svm_pred)
svm_accuracy = accuracy_score(Y_test, svm_pred)
svm_roc_auc = roc_auc_score(Y_test, svm_pred)

# Visualisaties

# Feature Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("Resultaten Tabel:")
print(results_table)
