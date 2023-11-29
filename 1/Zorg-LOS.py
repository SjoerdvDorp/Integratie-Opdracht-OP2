# author: Sjoerd van Dorp, 1032109, 14/11/2023, version 1.0
# Tussentijds opdracht 1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Load the 'train.csv' file
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Train data
X_train, Y_train = train_test_split(df, test_size=0.3, random_state=100)

# Informatie over de data set
print("Data Set Info:")
print(df.info())

# Omzetten van 'Stay' naar numerieke waarden
stay_mapping = {'0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35, '41-50': 45, '51-60': 55, '61-70': 65, '71-80': 75, '81-90': 85, '91-100': 95, 'More than 100 Days': 100}
df['Stay'] = df['Stay'].map(stay_mapping)

age_mapping = {'0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35, '41-50': 45, '51-60': 55, '61-70': 65, '71-80': 75, '81-90': 85, '91-100': 95, 'More than 100 Days': 100}
df['Age'] = df['Age'].map(age_mapping)

# Mapping van Hospital_type_code
hospital_type_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
df['Hospital_type_code'] = df['Hospital_type_code'].map(hospital_type_mapping)

# Mapping van Hospital_region_code
hospital_region_mapping = {'X': 1, 'Y': 2, 'Z': 3}
df['Hospital_region_code'] = df['Hospital_region_code'].map(hospital_region_mapping)

# Mapping van Department
department_mapping = {'anesthesia': 1, 'gynecology': 2, 'radiotherapy': 3, 'surgery': 4, 'TB & Chest disease': 5}
df['Department'] = df['Department'].map(department_mapping)

# Mapping van Ward_Type
ward_type_mapping = {'P': 1, 'Q': 2, 'R': 3, 'S': 4, 'T': 5, 'U': 6}
df['Ward_Type'] = df['Ward_Type'].map(ward_type_mapping)

# Mapping van Ward_Facility_Code
ward_facility_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
df['Ward_Facility_Code'] = df['Ward_Facility_Code'].map(ward_facility_mapping)

# Mapping van Type of Admission
admission_mapping = {'Emergency': 1, 'Trauma': 2, 'Urgent': 3}
df['Type of Admission'] = df['Type of Admission'].map(admission_mapping)

# Mapping van Severity of Illness
severity_mapping = {'Minor': 1, 'Moderate': 2, 'Extreme': 3}
df['Severity of Illness'] = df['Severity of Illness'].map(severity_mapping)

# Statistische Waarden in de data set
print("\nStatistische waarden Data Set:")
print(df.describe())

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

# Aantal ontbrekende waarden in de data set
print("\nOntbrekende waarden in Data Set:")
print(df.isnull().sum())

# Aanmaken van functies om lege velden te vullen
print("\nOntbrekende waarden in Data Set vullen: ...")
df['Stay'] = df['Stay'].fillna(df['Stay'].mode()[0])
df['Bed Grade'] = df['Bed Grade'].fillna(df['Bed Grade'].mode()[0])

# Ontbrekende waarden in de data set opnieuw printen
print("\nNieuwe ontbrekende waarden in Data Set:")
print(df.isnull().sum())

# Unieke waarden in de data set
print("\nUnieke waarden in Data Set:")
print(df.nunique())

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


# Plot voor "Hospital_type_code"
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Hospital_type_code"], palette="Blues")
plt.title("Hospital Type Code")
plt.xlabel("Hospital Type Code")
plt.ylabel("Count")
plt.show()

# Plot voor "Ward_Type"
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Ward_Type"], palette="Greens")
plt.title("Ward Type")
plt.xlabel("Ward Type")
plt.ylabel("Count")
plt.show()

# Plot voor "Bed Grade"
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Bed Grade"], palette="Reds")
plt.title("Bed Grade")
plt.xlabel("Bed Grade")
plt.ylabel("Count")
plt.show()

# Plot voor "Age"
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Age"], palette="Purples")
plt.title("Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Save the transformed data to 'Zorg-LOS.csv'
df.to_csv('Zorg-LOS.csv', index=False)
