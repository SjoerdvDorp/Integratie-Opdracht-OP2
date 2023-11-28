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

# Inlezen van de data set
data_set = pd.read_csv('Zorg-LOS CSV.csv')

# Train data
X_train, X_test = train_test_split(data_set, test_size=0.3, random_state=42)

# Informatie over de data set
print("Data Set Info:")
print(data_set.info())

# Definieer een mapping van stringwaarden naar float getallen
stay_mapping = {
    '0-10': 5,
    '11-20': 15,
    '21-30': 25,
    '31-40': 35,
    '41-50': 45,
    '51-60': 55,
    '61-70': 65,
    '71-80': 75,
    '81-90': 85,
    '91-100': 95,
    'More than 100 Days': 100,
}

# Convert 'Stay' kolom naar float met gebruik van mapping
data_set['Stay'] = data_set['Stay'].map(stay_mapping)

# Statistische Waarden in de data set
print("\nStatistische waarden Data Set:")
print(data_set.describe())

# Normaal verdeling plot
selected_column = 'Admission_Deposit'
plt.figure(figsize=(10, 6))
sns.histplot(data_set[selected_column], bins=30, kde=True, color='blue', stat='density')

mean_value, std_dev_value = data_set[selected_column].mean(), data_set[selected_column].std()
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
print(data_set.isnull().sum())

# Aanmaken van functies om lege velden te vullen
print("\nOntbrekende waarden in Data Set vullen: ...")
data_set['Stay'] = data_set['Stay'].fillna(data_set['Stay'].mode()[0])
data_set['Bed Grade'] = data_set['Bed Grade'].fillna(data_set['Bed Grade'].mode()[0])

# Ontbrekende waarden in de data set opnieuw printen
print("\nNieuwe ontbrekende waarden in Data Set:")
print(data_set.isnull().sum())

# Unieke waarden in de data set
print("\nUnieke waarden in Data Set:")
print(data_set.nunique())

# Selecteer alleen numerieke kolommen
numerical_columns = data_set.select_dtypes(include=[np.number])

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
sns.boxplot(data=data_set[columns_for_boxplot])

# Set plot title
plt.title('Boxplot')

# Show the plot
plt.show()

# Zet de kolommen om naar categorische variabelen
data_set["Hospital_type_code"] = data_set["Hospital_type_code"].astype("category")
data_set["Ward_Type"] = data_set["Ward_Type"].astype("category")
data_set["Bed Grade"] = data_set["Bed Grade"].astype("category")
data_set["Age"] = data_set["Age"].astype("category")


# Plot voor "Hospital_type_code"
plt.figure(figsize=(8, 5))
sns.countplot(x=data_set["Hospital_type_code"], palette="Blues")
plt.title("Hospital Type Code")
plt.xlabel("Hospital Type Code")
plt.ylabel("Count")
plt.show()

# Plot voor "Ward_Type"
plt.figure(figsize=(8, 5))
sns.countplot(x=data_set["Ward_Type"], palette="Greens")
plt.title("Ward Type")
plt.xlabel("Ward Type")
plt.ylabel("Count")
plt.show()

# Plot voor "Bed Grade"
plt.figure(figsize=(8, 5))
sns.countplot(x=data_set["Bed Grade"], palette="Reds")
plt.title("Bed Grade")
plt.xlabel("Bed Grade")
plt.ylabel("Count")
plt.show()

# Plot voor "Age"
plt.figure(figsize=(8, 5))
sns.countplot(x=data_set["Age"], palette="Purples")
plt.title("Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
