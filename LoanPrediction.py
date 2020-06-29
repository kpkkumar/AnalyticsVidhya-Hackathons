# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:04:28 2020

@author: inspiron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

traindata= pd.read_csv("D:\Studies\Books\Data Science\Python Projects\Analytics Vidhya\Loan Prediction\Dataset\TrainData.csv")
traindata.dtypes
traindata= pd.DataFrame(traindata)
traindata["Credit_History"].describe()
traindata = traindata.astype({"Credit_History" : "category"})
traindata["ApplicantIncome"].describe()
traindata = traindata.astype({"ApplicantIncome" : "int64"})
traindata = traindata.astype({"CoapplicantIncome" : "int64"})
traindata = traindata.astype({"LoanAmount" : "int64"})

traindata["Dependents"].describe()

traindata.isnull().sum()

traindata["Dependents"] = traindata["Dependents"].fillna("0")

traindata["Gender"].describe()
traindata["Gender"] = traindata["Gender"].fillna("Male")
traindata["Married"].describe()
traindata["Married"] = traindata["Married"].fillna("Yes")
traindata["Self_Employed"].describe()
traindata["Self_Employed"] = traindata["Self_Employed"].fillna("No")

traindata["LoanAmount"].describe()

traindata["LoanAmount"].fillna(traindata.LoanAmount.mean(), inplace=True)
traindata["Loan_Amount_Term"].describe()

traindata["Loan_Amount_Term"].fillna(traindata.Loan_Amount_Term.mean(), inplace=True)
traindata = traindata.astype({"Loan_Amount_Term" : "int64"})

traindata["Credit_History"].describe()
traindata["Credit_History"] = traindata["Credit_History"].fillna(1)

correlation = traindata.corr()

plot = sns.pairplot(traindata)
traindata = traindata.astype({"Married" : "category"})
traindata = traindata.astype({"Gender" : "category"})
traindata = traindata.astype({"Education" : "category"})
traindata = traindata.astype({"Self_Employed" : "category"})
traindata = traindata.astype({"Property_Area" : "category"})
traindata = traindata.astype({"Loan_Status" : "category"})

sns.countplot(x=traindata.Gender)
sns.countplot(x=traindata.Married)
sns.countplot(x=traindata.Dependents)
sns.countplot(x=traindata.Education)
sns.countplot(x=traindata.Self_Employed)
sns.countplot(x=traindata.Credit_History)
sns.countplot(x=traindata.Property_Area)
sns.countplot(x=traindata.Loan_Status)


from sklearn.utils import resample
major = traindata[traindata.Loan_Status == 'Y']
minor = traindata[traindata.Loan_Status == 'N']

minor_upsample = resample(minor, replace = True, n_samples = 422, random_state = 100)
train = pd.concat([major,minor_upsample])

sns.countplot(train.Loan_Status)
train.dtypes

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
train['Gender'] = en.fit_transform(train['Gender'])
train['Married'] = en.fit_transform(train['Married'])
train['Dependents'] = en.fit_transform(train['Dependents'])
train['Education'] = en.fit_transform(train['Education'])
train['Self_Employed'] = en.fit_transform(train['Self_Employed'])
train['Property_Area'] = en.fit_transform(train['Property_Area'])

sns.boxplot(traindata.LoanAmount)
sns.boxplot(traindata.Loan_Amount_Term)
sns.boxplot(traindata.ApplicantIncome)
sns.boxplot(traindata.CoapplicantIncome)

train.Loan_Status = train.Loan_Status.replace("Y" , "1")

train.Loan_Status = train.Loan_Status.replace("N" , "0")

sns.countplot(train.Loan_Status)

xtrain = train.drop('Loan_Status', axis = 1)
ytrain = train.Loan_Status
xtrain = xtrain.drop('Loan_ID', axis=1)

test=pd.read_csv("D:\Studies\Books\Data Science\Python Projects\Analytics Vidhya\Loan Prediction\Dataset\TestData.csv")
test.dtypes
test.isnull().sum()

test["Gender"].describe()
test["Gender"] = test["Gender"].fillna("Male")
test["Married"].describe()
test["Married"] = test["Married"].fillna("Yes")
test["Self_Employed"].describe()
test["Self_Employed"] = test["Self_Employed"].fillna("No")
test["Dependents"].describe()
test["Dependents"] = test["Dependents"].fillna("0")

test["LoanAmount"].describe()

test["LoanAmount"].fillna(traindata.LoanAmount.mean(), inplace=True)
test["Loan_Amount_Term"].describe()

test["Loan_Amount_Term"].fillna(traindata.Loan_Amount_Term.mean(), inplace=True)
test = test.astype({"Loan_Amount_Term" : "int64"})

test["Credit_History"].describe()
test["Credit_History"] = test["Credit_History"].fillna(1)

test = test.astype({"Married" : "category"})
test = test.astype({"Gender" : "category"})
test = test.astype({"Education" : "category"})
test = test.astype({"Self_Employed" : "category"})
test = test.astype({"Property_Area" : "category"})

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
test['Gender'] = en.fit_transform(test['Gender'])
test['Married'] = en.fit_transform(test['Married'])
test['Dependents'] = en.fit_transform(test['Dependents'])
test['Education'] = en.fit_transform(test['Education'])
test['Self_Employed'] = en.fit_transform(test['Self_Employed'])
test['Property_Area'] = en.fit_transform(test['Property_Area'])

test = test.drop('Loan_ID', axis=1)

from sklearn.ensemble import RandomForestRegressor
ranfor = RandomForestRegressor(n_estimators = 1300, random_state = 100)
ranfor.fit(xtrain,ytrain)

prediction = ranfor.predict(test)
prediction = pd.DataFrame(prediction)
prediction.to_csv(r'D:\\Studies\\Books\\Data Science\\Python Projects\\Analytics Vidhya\\Loan Prediction\\Dataset\\submission.csv')
