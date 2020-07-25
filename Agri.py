import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

train = pd.read_csv("D:\Studies\Books\Data Science\Python Projects\Analytics Vidhya\Agriculture\Dataset\Train.csv")

train.head()

train.dtypes

train['Crop_Type']= train['Crop_Type'].astype('category')

train['Soil_Type']=train['Soil_Type'].astype('category')

train['Season'] = train['Season'].astype('category')

train['Crop_Damage']=train['Crop_Damage'].astype('category')

train['Pesticide_Use_Category'] = train['Pesticide_Use_Category'].astype('category')

train.isnull().sum()

train.Number_Weeks_Used.mean()

train['Number_Weeks_Used']=train['Number_Weeks_Used'].fillna(value=28)

train['Number_Weeks_Used']=train['Number_Weeks_Used'].astype('int64')

correlation= train.corr()

correlation

train.describe()

sns.boxplot(train.Estimated_Insects_Count)

train=train[train.Estimated_Insects_Count<3500]

sns.boxplot(train.Number_Doses_Week)

train=train[train.Number_Doses_Week<66]

sns.boxplot(train.Number_Weeks_Used)

train=train[train.Number_Weeks_Used<60]

sns.boxplot(train.Number_Weeks_Quit)

train=train[train.Number_Weeks_Quit<40]

train.describe()

sns.countplot(x=train.Crop_Damage)

sns.countplot(x=train.Season)

sns.countplot(x=train.Pesticide_Use_Category)

sns.countplot(x=train.Crop_Type)

sns.countplot(x=train.Soil_Type)

table= pd.crosstab(train.Soil_Type, train.Crop_Damage)

table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = 'True')

table= pd.crosstab(train.Crop_Type, train.Crop_Damage)

table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = 'True')

table= pd.crosstab(train.Season, train.Crop_Damage)

table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = 'True')

table= pd.crosstab(train.Pesticide_Use_Category, train.Crop_Damage)

table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = 'True')
