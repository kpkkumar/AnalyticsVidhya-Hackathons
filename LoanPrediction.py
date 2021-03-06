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
traindata = traindata.astype({"LoanAmount" : "int64"})
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



pd.crosstab(traindata.Gender, traindata.Loan_Status)

table= pd.crosstab(traindata.Gender, traindata.Loan_Status)
table.div(table.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = 'True')

table= pd.crosstab(traindata.Married, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar', stacked ='True')

table= pd.crosstab(traindata.Dependents, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked='True')

table=pd.crosstab(traindata.Education, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked='True')

table=pd.crosstab(traindata.Self_Employed, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked='True')

table=pd.crosstab(traindata.Credit_History, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked='True')

table=pd.crosstab(traindata.Property_Area, traindata.Loan_Status)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked='True')


traindata=traindata.drop(['Loan_ID','Gender','Self_Employed'], axis=1)

sns.boxplot(traindata.ApplicantIncome)
traindata['ApplicantIncome'].describe()
traindata = traindata[(traindata.ApplicantIncome <=8500)]

sns.boxplot(traindata.Loan_Amount_Term)
traindata['Loan_Amount_Term'].describe()

sns.boxplot(traindata.LoanAmount)
traindata['LoanAmount'].describe()

from sklearn.utils import resample
major = traindata[traindata.Loan_Status == 'Y']
minor = traindata[traindata.Loan_Status == 'N']

minor_upsample = resample(minor, replace = True,n_samples=343,  random_state = 100)
train = pd.concat([major,minor_upsample])

sns.countplot(train.Loan_Status)
train.dtypes

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
train['Married'] = en.fit_transform(train['Married'])
train['Dependents'] = en.fit_transform(train['Dependents'])
train['Education'] = en.fit_transform(train['Education'])
train['Property_Area']=en.fit_transform(train['Property_Area'])


train.Loan_Status = train.Loan_Status.replace("Y" , "1")

train.Loan_Status = train.Loan_Status.replace("N" , "0")

sns.countplot(train.Loan_Status)

xtrain = train
xtrain = train.drop(['Loan_Status','CoapplicantIncome'], axis = 1)
ytrain = train.Loan_Status




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

test=test.drop(['Loan_ID','Gender','Self_Employed','CoapplicantIncome'], axis=1)


from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
test['Married'] = en.fit_transform(test['Married'])
test['Dependents'] = en.fit_transform(test['Dependents'])
test['Education'] = en.fit_transform(test['Education'])
test['Property_Area'] = en.fit_transform(test['Property_Area'])



'''from sklearn.ensemble import RandomForestRegressor
ranfor = RandomForestRegressor(n_estimators = 1000, random_state = 100)
ranfor.fit(xtrain,ytrain)'''

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)

prediction = logreg.predict(test)
prediction = pd.DataFrame(prediction)

prediction.to_csv(r'D:\\Studies\\Books\\Data Science\\Python Projects\\Analytics Vidhya\\Loan Prediction\\Dataset\\submission.csv')
