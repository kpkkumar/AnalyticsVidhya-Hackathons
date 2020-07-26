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

train['Number_Weeks_Used'].describe()

train['Number_Weeks_Used']=train['Number_Weeks_Used'].fillna(value=32)

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

train.Crop_Damage.value_counts()

train.columns


'''from sklearn.utils import resample
major = train[train.Crop_Damage==1]
minor = train[train.Crop_Damage==0]
minor_upsample1=resample(minor, replace=True, n_samples=11214 )


from sklearn.utils import resample
major = train[train.Crop_Damage==0]
minor = train[train.Crop_Damage==2]
minor_upsample2=resample(minor, replace=True, n_samples=71430 )


sample_train= pd.concat([major,minor_upsample1])

sample_train.Crop_Damage.value_counts()

sample_train.column'''

xtrain=train[['Crop_Type','Estimated_Insects_Count','Pesticide_Use_Category','Number_Doses_Week', 'Number_Weeks_Used','Number_Weeks_Quit']]

ytrain=train.Crop_Damage


test=pd.read_csv("D:\Studies\Books\Data Science\Python Projects\Analytics Vidhya\Agriculture\Dataset\Test.csv")

test.dtypes

test['Crop_Type']= test['Crop_Type'].astype('category')

test['Soil_Type']=test['Soil_Type'].astype('category')

test['Season'] = test['Season'].astype('category')

test['Pesticide_Use_Category'] = test['Pesticide_Use_Category'].astype('category')

test.isnull().sum()

test.Number_Weeks_Used.mean()

test.Number_Weeks_Used=test.Number_Weeks_Used.fillna(value=32)

testdata=test[['Crop_Type','Estimated_Insects_Count','Pesticide_Use_Category','Number_Doses_Week', 'Number_Weeks_Used','Number_Weeks_Quit']]

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(xtrain, ytrain)

prediction = logreg.predict(testdata)

prediction = pd.DataFrame(prediction)

prediction.to_csv(r'D:\Studies\Books\Data Science\Python Projects\Analytics Vidhya\Agriculture\Dataset\Output.csv')



