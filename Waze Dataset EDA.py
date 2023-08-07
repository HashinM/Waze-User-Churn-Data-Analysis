# This work was originally completed in a Jupyter Notebook


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset into a dataframe

df = pd.read_csv('waze_dataset.csv')

df.head()

df.size

df.describe()

df.info()



# The number of occurrence of a user opening the app during the month

# Box plot

plt.figure(figsize = (8,3))
plt.title('Sessions Box Plot')
sns.boxplot(df['sessions'])

# Histogram

plt.figure(figsize = (7,3))
plt.title('Sessions Histogram')
sns.histplot(df['sessions'])
median = df['sessions'].median()
plt.axvline(median, color = 'red', linestyle='--')
plt.text(median,1200, 'median={}'.format(median), color = 'red')

## Creating a function to aid in the plotting of histograms

def histogram(column_str, median_text=True, **kwargs):
    
    median=round(df[column_str].median(),1)
    plt.figure(figsize=(7,3))
    plt.title(f'{column_str} histogram');
    ax = sns.histplot(x=df[column_str], **kwargs)
    plt.axvline(median, color='red', linestyle='--')
    if median_text == True:
        ax.text(0.25,0.85, f'median={median}', color='red',
            ha='left',va='top', transform=ax.transAxes)
    else: 
        print('Median:', median)

# An occurrence of driving at least 1 km during the month

# Box plot

plt.figure(figsize = (8,3))
plt.title('Drives Box Plot')
sns.boxplot(df['drives'])

# Histogram

histogram('drives')


# A model estimate of the total number of sessions since a user has onboarded

# Box plot

plt.figure(figsize=(8,3))
plt.title('Total Sessions Box Plot')
sns.boxplot(df['total_sessions'])

# Histogram

histogram('total_sessions')

# The number of days since a user signed up for the app

# Box plot

plt.figure(figsize=(8,3))
plt.title('Days Since Onboarding Box Plot')
sns.boxplot(df['n_days_after_onboarding'])

# Histogram

histogram('n_days_after_onboarding', median_text=False)


# Total kilometers driven during the month

# Box plot

plt.figure(figsize=(8,3))
plt.title('Total Kilometers Box Plot')
sns.boxplot(df['driven_km_drives'])

# Histogram

histogram('driven_km_drives')

# Total duration driven in minutes during the month

# Box plot

plt.figure(figsize=(8,3))
plt.title('Total Duration in Minutes Box Plot')
sns.boxplot(df['duration_minutes_drives'])

# Histogram

histogram('duration_minutes_drives')

# Number of days the user opens the app during the month

# Box plot

plt.figure(figsize=(8,3))
plt.title('Activity Days Box Plot')
sns.boxplot(df['activity_days'])

# Histogram

histogram('activity_days', median_text=False)

# Number of days the user drives (at least 1 km) during the month

# Box plot

plt.figure(figsize=(8,3))
plt.title('Driving Days Box Plot')
sns.boxplot(df['driving_days'])

# Histogram

histogram('driving_days', median_text=False)

# The type of device a user starts a session with

# Pie chart


data = df['device'].value_counts()
device_labels = ['iPhone', 'Android']
plt.figure(figsize=(4,4))
plt.title('User Device')
plt.pie(data, labels = device_labels, autopct = '%1.2f%%')

# If the user was labelled as 'Retained' or 'Churned'

# Pie chart

data = df['label'].value_counts()
label_labels = ['Retained', 'Churned']
plt.figure(figsize=(4,4))
plt.title('Retained vs Churned')
plt.pie(data, labels = label_labels, autopct = '%1.2f%%')

# Histogram that for each day has a bar representing the counts of `driving_days` and `user_days`

plt.figure(figsize=(12,5))
plt.title('Driving Days vs Activity Days Histogram')
variable_label = ['Driving Days', 'Activity Days']
plt.xlabel('Days')
plt.ylabel('Count')
plt.hist([df['driving_days'], df['activity_days']], bins = range(0,33), label = variable_label)
plt.legend()

# Confirming the maximum amount of days per 'driving_days' and 'activity_days' variables

print(df['driving_days'].max())
print(df['activity_days'].max())

# Plotting a scatter plot to validate the validity of these two variables

plt.title('Driving Days vs Activity Days Scatterplot')
sns.scatterplot(data=df, x='driving_days', y='activity_days')
plt.plot([0,31],[0,31], color = 'red', linestyle = '--')

# Histogram to show each device-label combination

plt.figure(figsize=(5,4))
plt.title('Retention by Device Histogram')
sns.histplot(data = df, x = 'device', hue = 'label', multiple = 'dodge', shrink = 0.8)


# Creating `km_per_driving_day` column

df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['km_per_driving_day'].describe()

# Converting infinite values to zero

df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# Confirming that it worked

df['km_per_driving_day'].describe()

# Histogram showing the churn rate by mean kilometers per driving day but disregarding any values greater than
# 1200km because these values are very unrealistic

plt.figure(figsize=(10,4))
plt.title('Churn Rate by Mean Kilometers per Driving Days Histogram')
sns.histplot(data = df, bins = range(0,1201,10), x = 'km_per_driving_day', hue = 'label', multiple = 'fill')
plt.ylabel('%', rotation = 0)

# Histogram of churn rate per driving day

plt.figure(figsize=(10,4))
plt.title('Churn Rate per Driving Days Histogram')
sns.histplot(data = df, bins = range(0,32), x = 'driving_days', hue = 'label', multiple = 'fill', discrete = True)
plt.ylabel('%', rotation = 0)

# Creating new column 'percent_sessions_in_last_month'

df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# Finding median value
df['percent_sessions_in_last_month'].median()

# Histogram of 'percent_sessions_in_last_month'

histogram('percent_sessions_in_last_month', hue = df['label'], multiple = 'layer', median_text = False)

# Checking the median value of the `n_days_after_onboarding` variable.

df['n_days_after_onboarding'].median()

# Making a histogram of `n_days_after_onboarding` for just the people who had 40% or more of their total sessions in the last month.

data = df.loc[df['percent_sessions_in_last_month']>=0.4]

plt.figure(figsize=(7,3))
plt.title('Days After Onboarding for Users With > 40% of Their Sessions in the Last Month')
sns.histplot(data['n_days_after_onboarding'])

# Creating a function to help deal with outliers

def outlier_imputation(column_name, percentile):
    
    threshold = df[column_name].quantile(percentile)
    df.loc[df[column_name] > threshold, column_name] = threshold
    print('{:25} | percentile: {}| threshold: {}'.format(column_name, percentile, threshold))


# Dealing with the outliers

for column in ['sessions', 'drives', 'total_sessions', 
               'driven_km_drives', 'duration_minutes_drives']:
                outlier_imputation(column, 0.95)

# Checking if the changes worked

df.describe()


# Creating a monthly drives per session column

df['monthly_drives_per_session'] = df['drives'] / df['sessions']

df.head()
