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
