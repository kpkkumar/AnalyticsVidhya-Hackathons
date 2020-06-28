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
traindata["Credit_History"].describe()
traindata = traindata.astype({"Credit_History" : "category"})
