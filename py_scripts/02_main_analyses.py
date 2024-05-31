# Emily & Prab
# Project 5: Hackathon
# DSB-318
# May 31, 2024
# Restaurant Revenue Predictions


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Working Directory
os.getcwd()
os.chdir(
  'C:/Users/emily/Git_Stuff/General_Assembly/04_Projects/project-5-cyoa-hackathon')
os.listdir()

# Data 
data = pd.read_csv('./original_data/train.csv')
val_data = pd.read_csv('./original_data/test.csv')

# Data Cleaning
data.isna().sum().sum() # 0, hooray!
data.info()

# Column names
col_names = [c.lower().replace(' ', '_') for c in list(data.columns)]
data.columns = col_names

val_col_names = [c.lower().replace(' ', '_') for c in list(val_data.columns)]
val_data.columns = val_col_names

# Dtype conversions
data['open_date'] = pd.to_datetime(data['open_date'])

# Are these worth including?
data['city'].nunique() # 37 is too many for the number of rows we have
data['city_group'].nunique()
data['type'].nunique()

data['city'].unique()
data['city_group'].unique()
data['type'].unique()
# FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
# inline = in store, like a freestanding one
# mobile = burger truck or something (we don't have any of these)

set(data['city'].unique())==set(val_data['city'].unique())
# False.  There are like 20 cities in the val data not present in the 
# regular data.  This means that even if we did OHE the data, it would
# have limited utility in the val_data.

# The city column is not useful for us
# data_backup = data.copy()  # make a copy, we're going to drop city for modelling
data.drop(columns = 'city', inplace = True)

# What are the distributions here?
data['city_group'].value_counts(dropna = False) #fine
data['type'].value_counts(dropna = False) # bad!
val_data['type'].value_counts(dropna = False) # other

# Binarize type because of weird proportions
data['fc_type'] = [1 if t=='FC' else 0 for t in data['type']]
data[['type', 'fc_type']] #check
# Drop the old column
data.drop(columns = 'type', inplace = True)

# Binarize city_group because why import OHE and all that for just this
data['city_group_ohe'] = [1 if t=='Big Cities' else 0 for t in data['city_group']]
data[['city_group', 'city_group_ohe']] #check
# Drop the old column
data.drop(columns = 'city_group', inplace = True)

# Check this again
data.info()

# Visualizations
autoplots(data, 'revenue')

# lr won't take a datetime
data['year'] = data['open_date'].dt.year
data['month'] = data['open_date'].dt.month

data.info() # awesome, they're just integers

# Drop the old column
data.drop(columns = 'open_date', inplace = True)

# Train Test Split
# With more time, we might have checked each column in case 
# we needed to stratify, but we have short time and it's already
# weird, so we're just plowing ahead.
X = data.drop(columns = ['revenue', 'id'])
y = data['revenue']

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.2, random_state = 19)  
# 3 rings for the elven kings
# 7 for the dwarf lords in their halls of stone
# 9 for mortal men doomed to die
  
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
lr.score(X_test, y_test)

# Well that was terrible.  Let's try regularization
ss = StandardScaler()
a = np.linspace(0.1, 50, 200)
lasso = LassoCV()
pipe = Pipeline([('ss', ss), ('lasso', lasso)])
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)

# Coefficients??
list(zip(X_train.columns, lasso.coef_))
# Good grief, everything is 0 except year, 
# and year is negative.

# SLRs
# year
lr_year = lr.fit(X_train[['year']], y_train)
lr_year.score(X_train[['year']], y_train)
lr_year.score(X_test[['year']], y_test)
lr_year.coef_

# food court or not
logr = LogisticRegression()
logr_fc = logr.fit(X_train[['fc_type']], y_train)
logr_fc.score(X_train[['fc_type']], y_train)
logr_fc.score(X_test[['fc_type']], y_test)

# Get some predictions
# full model
pipe_preds_X_train = pipe.predict(X_train)
pipe_preds_X_test = pipe.predict(X_test)

# SLR
lr_year_preds_X_train = lr_year.predict(X_train[['year']])
lr_year_preds_X_test = lr_year.predict(X_test[['year']])

# RMSE
print(f'''
LASSO model, training set: {mean_squared_error(y_train, pipe_preds_X_train)}
LASSO model, testing set: {mean_squared_error(y_test, pipe_preds_X_test)}
Year only, training set: {mean_squared_error(y_train, lr_year_preds_X_train)}
Year only, testing set: {mean_squared_error(y_test, lr_year_preds_X_test)}''')






