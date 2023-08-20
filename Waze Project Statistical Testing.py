import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('waze_dataset.csv')


# Creating map_dictionary

map_dictionary = {'iPhone': 1, 'Android': 2}

# Creating new device_type column

df['device_type'] = df['device']

# Map the new column to the dictionary

df['device_type'] = df['device_type'].map(map_dictionary)

# Check if the changes are correct
df.head()

# Average number of drives for each device type

df.groupby('device_type')['drives'].mean()

# Hypothesis testing
# Significane level = 5%

iPhone = df[df['device_type'] == 1]['drives']

Android = df[df['device_type'] == 2]['drives']

stats.ttest_ind(a = iPhone, b = Android, equal_var = False)

#   Since our P value (0.14) is larger than the significance level (0.05) we FAIL to reject the null hypothesis and deem that there is not a statistically significant difference in the average number of drives between iPhone and Android users
