# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:09:05 2019

@author: lopezf
"""

#use linear regression to predict values-----------------------
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


#pull the data------------------------------------------------------------------------------
#quandl.ApiConfig.api_key = 'gXgcc7a9mHYgochSzsuX'
#df = quandl.get('WIKI/GOOGL')
#df.head()
#df.to_csv("stock_data.csv",index = False)

df = pd.read_csv("stock_data.csv")


#grab some features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#feature engineering-----------------------------------------------------------------------

#now pull columns we care about (our features)
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
df.head()

#fill the missing
df.fillna(-99999, inplace = True)

#math.ceil will take anything and get to the ceiling so round up 
forecast_out = int(math.ceil(0.01*len(df)))

#shift custom % from the Adj.Close to get the prediction in the future
forecast_col = 'Adj. Close'
df['label'] = df[forecast_col].shift(-forecast_out)

#df.head()
#df.dropna(inplace = True)
#df.tail()

#training and testing---------------------------------------------------------------------

#x are your features, everything but your label column
x = np.array(df.drop(['label'],1))
#preprocess data and drop nas, repull y for labels
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]



df.dropna(inplace = True)
y = np.array(df['label'])
y = np.array(df['label'])
print(len(x),len(y))

#start building train and test
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = .2)

#define classifier, linear
clf = LinearRegression()
clf.fit(x_train,y_train) #train data
accuracy = clf.score(x_test,y_test) #test against the test data

#start predicting
forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy,forecast_out)