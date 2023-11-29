# author: Sjoerd van Dorp, 1032109, 29/11/2023, version 1.0
# Tussentijdse Opdracht 2

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
file_path = "C:\\Users\\sjoer\\OneDrive\\Documents\\HR\\Jaar 3\\OP2\\Predictive Analytics\\Tussentijdse opdrachten\\1\\train.csv"
df = pd.read_csv(file_path)

# Train data
X_train, Y_train = train_test_split(df, test_size=0.3, random_state=100)

# Informatie over de data set
print("Data Set Info:")
print(df.info())