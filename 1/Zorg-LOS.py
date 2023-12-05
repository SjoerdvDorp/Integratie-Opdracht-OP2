# author: Sjoerd van Dorp, 1032109, 14/11/2023, version 4.0
# Tussentijdse Opdracht 1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'train.csv' file
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Informatie over de data set
print("Data Set Info:")
print(df.info())

# Drop onnodige kolommen
print("\nOnnodige kolommen droppen...")
columns_to_drop = ['case_id', 'patientid']
df = df.drop(columns=columns_to_drop)

le = LabelEncoder()
df['Stay'] = le.fit_transform(df['Stay'].astype('str'))
# print(traindf.head())

#Label Encoding all the columns in Train and test datasets
for i in ['Hospital_type_code', 'Hospital_region_code',
'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype(str))

# Replacing NA values in Bed Grade Column for Train datasets
df['Bed Grade'].fillna(df['Bed Grade'].mode()[0],
                            inplace=True)
# Replacing NA values in Column for Train datasets
df['City_Code_Patient'].fillna(df['City_Code_Patient'].
                                    mode()[0], inplace=True)

# Label Encoding Stay column in train dataset
from sklearn.preprocessing import LabelEncoder

# Statistische Waarden in de data set
print("\nStatistische waarden Data Set:")
print(df.describe())

# Aantal ontbrekende waarden in de data set
print("\nOntbrekende waarden in Data Set:")
print(df.isnull().sum())

# Aanmaken van functies om lege velden te vullen
print("\nOntbrekende waarden in Data Set vullen: ...")
df['Bed Grade'] = df['Bed Grade'].fillna(df['Bed Grade'].mode()[0])

# Unieke waarden in de data set
print("\nUnieke waarden in Data Set:")
print(df.nunique())

# Normaal verdeling plot
selected_column = 'Admission_Deposit'
plt.figure(figsize=(10, 6))
sns.histplot(df[selected_column], bins=30, kde=True, color='blue', stat='density')

mean_value, std_dev_value = df[selected_column].mean(), df[selected_column].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_value, std_dev_value)
plt.plot(x, p, 'k', linewidth=2)

plt.title(f'Normale Verdeling van {selected_column}')
plt.xlabel('Waarden')
plt.ylabel('Frequentie')
plt.show()

# Selecteer alleen numerieke kolommen
numerical_columns = df.select_dtypes(include=[np.number])

# Creëer een correlatiematrix van numerieke kolommen
correlation_matrix = numerical_columns.corr()

# Creëer een heatmap met seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Kolommen voor de boxplot
columns_for_boxplot = ['Available Extra Rooms in Hospital', 'Visitors with Patient']

# Create a boxplot for selected columns
sns.boxplot(data=df[columns_for_boxplot])

# Set plot title
plt.title('Boxplot')

# Show the plot
plt.show()

# Zet de kolommen om naar categorische variabelen
df["Hospital_type_code"] = df["Hospital_type_code"].astype("category")
df["Ward_Type"] = df["Ward_Type"].astype("category")
df["Bed Grade"] = df["Bed Grade"].astype("category")
df["Age"] = df["Age"].astype("category")

# Plot voor "Age"
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Age"], palette="Purples")
plt.title("Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Save the transformed data to 'Zorg-LOS.csv'
df.to_csv('Zorg-LOS.csv', index=False)
