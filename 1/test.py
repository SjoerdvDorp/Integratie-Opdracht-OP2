#Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # to plot charts
from datetime import date
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Import datasets
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# Print dataset information
print(traindf.info())
print("\n\n")
print(traindf.describe())
print(traindf.head())

# Data Overview
print("\n")

# Explore missing values
print("Missing values in dataset:")
print(traindf.isnull().sum())
print(traindf.isnull().sum().sort_values(ascending = False))
print(traindf.Stay.unique())
print("\n")
print(traindf.shape)
print("\n")

# Number of distinct observations in train dataset
for i in traindf.columns:
    print(i, ':', traindf[i].nunique())

# Data Preparation

# Drop onnodige kolommen
print("\nOnnodige kolommen droppen...")
columns_to_drop = ['case_id', 'patientid']
traindf = traindf.drop(columns=columns_to_drop)
testdf = testdf.drop(columns=columns_to_drop)

#Replacing NA values in Bed Grade Column for both Train and Test datssets
traindf['Bed Grade'].fillna(traindf['Bed Grade'].mode()[0], inplace = True)
testdf['Bed Grade'].fillna(testdf['Bed Grade'].mode()[0], inplace = True)

#Replacing NA values in  Column for both Train and Test datssets
traindf['City_Code_Patient'].fillna(traindf['City_Code_Patient'].mode()[0], inplace = True)
testdf['City_Code_Patient'].fillna(testdf['City_Code_Patient'].mode()[0], inplace = True)

# Label Encoding Stay column in train dataset
le = LabelEncoder()
traindf['Stay'] = le.fit_transform(traindf['Stay'].astype('str'))
print("\n\n")
print(traindf.head())

#Imputing dummy Stay column in test datset to concatenate with train dataset
testdf['Stay'] = -1
df = pd.concat([traindf, testdf])
df.shape

#Label Encoding all the columns in Train and test datasets
for i in ['Hospital_type_code', 'Hospital_region_code', 'Department',
          'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype(str))

#Spearating Train and Test Datasets
traindf = df[df['Stay']!=-1]
testdf = df[df['Stay']==-1]

print("\n")
print(traindf.head())

# Create a Headmap
plt.figure(figsize=(13,10))
sns.heatmap(traindf.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

# Splitting train data for Naive Bayes and XGBoost
X1 = traindf.drop('Stay', axis=1)
y1 = traindf['Stay']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size =0.20, random_state =100)
print("\n")
print(X_train.head())

# Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
target = y_train.values
features = X_train.values
classifier_nb = GaussianNB()
model_nb = classifier_nb.fit(features, target)
prediction_nb = model_nb.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score_nb = accuracy_score(prediction_nb,y_test)
print("\nNaive Bayes Acurracy:", acc_score_nb*100)

# XGBoost Model
import xgboost
classifier_xgb = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1,
n_estimators=800,objective='multi:softmax', reg_alpha=0.5,
reg_lambda=1.5, booster='gbtree', n_jobs=4, min_child_weight=2, base_score= 0.75)
model_xgb = classifier_xgb.fit(X_train, y_train)
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = accuracy_score(prediction_xgb,y_test)
print("\n XGboost Accuracy:", acc_score_xgb*100)

# LazyClassifier Instance and fitting data
#cls = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
#models, predictions = cls.fit(X_train, X_test, y_train, y_test)
#print(models.to_string())
# Choose a specific model from the trained models (for example, the first model)
#chosen_model = models.iloc[0]
# Get the model name
#model_name = chosen_model[0]
# Get the actual model instance
#model_instance = chosen_model[1]
# Now, you can use the chosen model to make predictions on new data
#new_data_predictions = model_instance.predict(new_data)

print("Can I get better results?")

# Spilt the data in train and test
# Splitting train data for Naive Bayes and XGBoost
X1 = traindf.drop('Stay', axis=1)
y1 = traindf['Stay']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size =0.20, random_state =100)

# Outlier Detection
def detect_outliers(df, n, features):
    outlier_indices = []
    """
    Detect outliers from given list of features. It returns a list of the indices
    according to the observations containing more than n outliers according
    to the Tukey method
    """
    # iterate over features(columns)
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

# detect outliers from numeric features
outliers_to_drop = detect_outliers(df, 2, ["Hospital_code", 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital',
                                           'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness',
                                           'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay'])
# Drop outliers
df.drop(df.loc[outliers_to_drop].index, inplace=True)
# Modeling
# Data Transformation
q  = QuantileTransformer()
X = q.fit_transform(df)
transformedDF = q.transform(X)
transformedDF = pd.DataFrame(X)
transformedDF.columns =["Hospital_code", 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital'
                                                                                                             '',
                                           'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness',
                                           'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay']
# Show top 5 rows
print(transformedDF.head())
# Split data in test and train dataset. Train dataset will be used in Model training and evaluation and test dataset will be used in prediction.
# Before i predict the test data, i performed cross validation for various models. C
# Code splits dataset into train (70%) and test (30%) dataset.
features = df.drop(["Stay"], axis=1)
labels = df["Stay"]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=7)
