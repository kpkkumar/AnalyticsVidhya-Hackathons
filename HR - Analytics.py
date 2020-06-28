# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:33:27 2020

@author: inspiron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("D:\\Studies\\Books\\Data Science\\Python Projects\\Analytics Vidhya\\HR - Promotion\\Datasets\\train_LZdllcl.csv")
train.dtypes
train = train.astype({"is_promoted" : 'category'})
train = train.astype({"awards_won?":'category'})
train = train.astype({"KPIs_met >80%":'category'})
train["previous_year_rating"] = train["previous_year_rating"].fillna(6)
train = train.astype({"previous_year_rating":'category'})

train.dtypes
train.isnull().sum()
train.education.describe()
train["education"] = train["education"].fillna("Bachelor's")
train.education.describe()
train.isnull().sum()
train.previous_year_rating.describe()
train["previous_year_rating"] = train["previous_year_rating"].replace({6:3})
train.previous_year_rating.describe()
train = train.astype({"previous_year_rating" : 'category'})

corr = train.corr()


sns.boxplot(x=train.avg_training_score)
sns.boxplot(x=train.age)
train = train[(train.age <= 51)]
train.is_promoted.value_counts()
 
from sklearn.utils import resample

major = train[train.is_promoted == 0]
minor = train[train.is_promoted == 1]

minor_upsample = resample(minor, replace = True, n_samples = 47834, random_state = 100)
upsample = pd.concat([major,minor_upsample])

upsample.is_promoted.value_counts()
ytrain = upsample.is_promoted
ytrain = pd.DataFrame(ytrain).astype("category")
upsample.columns
xtrain = upsample.drop('is_promoted', axis =1)
xtrain.dtypes

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
xtrain['department'] = en.fit_transform(xtrain['department'])
xtrain['region'] = en.fit_transform(xtrain['region'])
xtrain['department'] = en.fit_transform(xtrain['department'])
xtrain['education'] = en.fit_transform(xtrain['education'])
xtrain['gender'] = en.fit_transform(xtrain['gender'])
xtrain['recruitment_channel'] = en.fit_transform(xtrain['recruitment_channel'])

xtrain = xtrain.drop('employee_id',axis=1)
xtrain = xtrain.drop('age', axis =1)
'''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)'''

test = pd.read_csv("D:\\Studies\\Books\\Data Science\\Python Projects\\Analytics Vidhya\\HR - Promotion\Datasets\\test_2umaH9m.csv")
test.dtypes
test.isnull().sum()
test.education.describe()
test['education']=test['education'].fillna("Bachelor's")
test.previous_year_rating.value_counts()
test['previous_year_rating'] = test['previous_year_rating'].fillna(3)
test = test.astype({"department":'category'})
test = test.astype({"region":'category'})
test = test.astype({"education":'category'})
test = test.astype({"gender":'category'})
test = test.astype({"recruitment_channel":'category'})
test = test.astype({"previous_year_rating":'category'})
test.dtypes


test['department'] = en.fit_transform(test['department'])
test['region'] = en.fit_transform(test['region'])
test['department'] = en.fit_transform(test['department'])
test['education'] = en.fit_transform(test['education'])
test['gender'] = en.fit_transform(test['gender'])
test['recruitment_channel'] = en.fit_transform(test['recruitment_channel'])
test = test.drop('employee_id', axis=1)
test = test.drop('age',axis=1)
'''prediction = model.predict(test)
prediction = pd.DataFrame(prediction)'''

from sklearn.ensemble import RandomForestRegressor
ranfor = RandomForestRegressor(n_estimators = 1300, random_state = 100)
ranfor.fit(xtrain,ytrain)


prediction = ranfor.predict(test)
prediction = pd.DataFrame(prediction)
prediction.to_csv(r'D:\\Studies\\Books\\Data Science\\Python Projects\\Analytics Vidhya\\HR - Promotion\\Datasets\\submission.csv')
