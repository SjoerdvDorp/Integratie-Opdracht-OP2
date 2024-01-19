# author: Sjoerd van Dorp, 1032109, 02/01/2024, version 1.0
# Integratie Opdracht

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

#Opdracht 1
#---------------------------------------------------------

# Import van datasets
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# Print dataset informatie
print("\nDataset informatie:")
print(traindf.info())
print("\nStatistische waarden:")
print(traindf.describe())
print("\nOntbrekende waarden in Data Set:")
print(traindf.isnull().sum())

# Aantal unieke waarden
print("\nUnieke waarden in Data Set:")
for i in traindf.columns:
    print(i, ':', traindf[i].nunique())

# Data Preparation

# Drop onnodige kolommen
print("\nOnnodige kolommen droppen...")
columns_to_drop = ['case_id', 'patientid']
traindf = traindf.drop(columns=columns_to_drop)
testdf = testdf.drop(columns=columns_to_drop)

# NA waarden in de Bed Grade kolom voor de Train en Test datasets vervangen met de modus
print("\nLege waarden in 'Bed Grade' en 'City_Code_Patient' verwijderen...")
traindf['Bed Grade'].fillna(traindf['Bed Grade'].mode()[0], inplace=True)
testdf['Bed Grade'].fillna(testdf['Bed Grade'].mode()[0], inplace=True)

# NA waarden in de City_Code_Patient kolom voor de Train en Test datasets vervangen met de modus
traindf['City_Code_Patient'].fillna(traindf['City_Code_Patient'].mode()[0], inplace = True)
testdf['City_Code_Patient'].fillna(testdf['City_Code_Patient'].mode()[0], inplace = True)

# Labelcodering van de kolom 'Stay' in de trainingsdataset
print("\nTekst naar numerieke waarden omzetten in train dataset...")
le = LabelEncoder()
traindf['Stay'] = le.fit_transform(traindf['Stay'].astype('str'))

# Het invoegen van een dummy 'Stay'-kolom in de testdataset om samen te voegen met de trainingsdataset.
print("\nDummy waarden in Stay kolom zetten...")
testdf['Stay'] = -1
df = pd.concat([traindf, testdf])
df.shape

# Labelcodering toepassen op alle kolommen in de trainings- en testdatasets.
print("\nTekst naar numerieke waarden omzetten in train en test dataset...")
for i in ['Hospital_type_code', 'Hospital_region_code', 'Department',
          'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype(str))

print("\nOntbrekende waarden in Data Set:")
print(traindf.isnull().sum())

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

print("\n\nDe modellen worden nu getraind en de accuracy word geprint: ")

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
classifier_xgb = XGBClassifier(max_depth=4, learning_rate=0.1,
n_estimators=800,objective='multi:softmax', reg_alpha=0.5,
reg_lambda=1.5, booster='gbtree', n_jobs=4, min_child_weight=2, base_score= 0.75)
model_xgb = classifier_xgb.fit(X_train, y_train)
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = accuracy_score(prediction_xgb,y_test)
print("\nXGboost Accuracy:", acc_score_xgb*100)

# Vervang de numerieke labels door de werkelijke categorieën voor de voorspellingen op X_test voor XGBoost
prediction_xgb_categories = pd.Series(prediction_xgb).replace({0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})

# Toon de resultaten
print("\nAantal voorspellingen per categorie in het train model van XGBoost:")
print(prediction_xgb_categories.value_counts())

# RandomForest Model
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=100)
model_rf = classifier_rf.fit(X_train, y_train)
prediction_rf = model_rf.predict(X_test)
acc_score_rf = accuracy_score(prediction_rf, y_test)
print("\nRandom Forest Accuracy:", acc_score_rf * 100)

print("\n\nNu voer ik de modeltest uit op de test dataset: ")

# Verwijder de 'Stay' kolom voor de voorspelling
testdf = testdf.drop('Stay', axis=1)

# Voer de voorspelling uit
pred_xgb = classifier_xgb.predict(testdf)
result_xgb = pd.DataFrame(pred_xgb, columns=['Stay'])

# Vervang de numerieke labels door de werkelijke categorieën
result_xgb['Stay'] = result_xgb['Stay'].replace({0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})

# Toon de resultaten
print("\nAantal voorspellingen per categorie in het test model:\n")
print(result_xgb['Stay'].value_counts())


print("\n\nLaten we nu voor dit model een confusion matrix en AOC maken... ")

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

print("\n\nDe validatietest: ")

# De kolom 'Stay' laat vallen
original_stay_labels = traindf['Stay'].copy()

# Selecteer voorspellende kolommen - met uitzondering van identificatiegegevens en de doelvariabele 'Stay'
predictor_columns = [col for col in traindf.columns if col not in ['case_id', 'patientid', 'Stay']]

# Categorische variabelen coderen in de trainingset
X_train = pd.get_dummies(traindf[predictor_columns], drop_first=True)

# Zorg ervoor dat X_train en originele_stay_labels hetzelfde aantal rijen hebben
assert X_train.shape[0] == original_stay_labels.shape[0], "Mismatch in number of rows after encoding"

# De trainingsgegevens opsplitsen in trainings- en validatiesets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, original_stay_labels, test_size=0.2, random_state=42)

# Maak de XGBoost-classificator en pas deze toe
xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(X_train_split, y_train_split)

# Voorspellen op de validatieset
y_val_pred = xgb_classifier.predict(X_val)

# Converteer de werkelijke labels naar categorieën
y_val_categories = y_val.replace({0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})

# Converteer de voorspelde labels naar categorieën
y_val_pred_categories = pd.Series(y_val_pred).replace({0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})

# De classifier evalueren met de omgezette categorieën
print("\nClassifier Report:\n", classification_report(y_val_categories, y_val_pred_categories))

