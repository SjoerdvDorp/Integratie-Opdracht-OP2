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

# Set the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'Zorg-LOS.csv' file
file_path = "C:\\Users\\sjoer\\OneDrive\\Documents\\HR\\Jaar 3\\OP2\\Predictive Analytics\\Tussentijdse opdrachten\\1\\Zorg-LOS.csv"
df = pd.read_csv(file_path)

# Select features and target variable
X = df.drop(columns=['Stay'])  # Remove the target variable 'Stay'
Y = df['Stay']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=10)

# Aantal ontbrekende waarden in de data set
print("\nOntbrekende waarden in Data Set:")
print(df.isnull().sum())

# Modellen testen

# Random Forest
rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=1)
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
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Print results
print("Random Forest Results:")
print(f"Confusion Matrix:\n{rf_conf_matrix}")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"AUC: {rf_roc_auc:.4f}")
print(classification_report(Y_test, rf_pred))

print("\nLogistic Regression Results:")
print(f"Confusion Matrix:\n{lr_conf_matrix}")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"AUC: {lr_roc_auc:.4f}")
print(classification_report(Y_test, lr_pred))

print("\nSupport Vector Machine Results:")
print(f"Confusion Matrix:\n{svm_conf_matrix}")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"AUC: {svm_roc_auc:.4f}")
print(classification_report(Y_test, svm_pred))
