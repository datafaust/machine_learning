# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:43:28 2019

@author: lopezf
"""

#for run in R, copy and paste into console without hash
#library(reticulate)
#use_virtualenv("root") #name the enviornment from conda to use
#repl_python()

#linear regressions from sentdex lesson-------------------------------------------------------------------------------
import os
import pandas as pd
import quandl, math
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression


#pull the data------------------------------------------------------------------------------
# quandl.ApiConfig.api_key = 'gXgcc7a9mHYgochSzsuX'
# df = quandl.get('WIKI/GOOGL')
# df.head()
# #write out then hash above
# df.to_csv("data/stock_data.csv", index = False)

#read data 
df = pd.read_csv("data/stock_data.csv")
 
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

df.head()
df.dropna(inplace = True)
df.tail()

#training and testing---------------------------------------------------------------------

#x are your features, everything but your label column
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

#preprocess data and drop nas, repull y for labels
x = preprocessing.scale(x)
df.dropna(inplace = True)
y = np.array(df['label'])
print(len(x),len(y))

#start building train and test
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = .2)

#define classifier, linear
clf = LinearRegression()
clf.fit(x_train,y_train) #train data
accuracy = clf.score(x_test,y_test) #test against the test data

print(accuracy)

#what if we wanted to use SVM regressions
clf = svm.SVR()
clf.fit(x_train,y_train) #train data
accuracy = clf.score(x_test,y_test) #test against the test data

print(accuracy)

#what if we wanted to use SVM regression with a different kernal?
clf = svm.SVR(kernel = "poly")
clf.fit(x_train,y_train) #train data
accuracy = clf.score(x_test,y_test) #test against the test data

print(accuracy)






 
