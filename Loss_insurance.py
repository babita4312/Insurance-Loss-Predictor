# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:59:01 2019

@author: Babita

Task B: 1) Predict the loss amount for the insurance policies using the historical trends and features. 
2) You can use any programming language to do this.
3) Create a sample data set for implementing the feature.
4) Drive information by analyzing the data set you have made and present valuable information from it in a presentable format.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
#reading csv file
df= pd.read_csv('insurance.csv')
df.replace('-999.25','0.0',inplace=True)
columns=('age','bmi','children','smoker','charges')
df1=pd.DataFrame(df,columns=columns)
#generate data
X=np.array(df1.drop(['age'],1))
y=np.array(df1['age'])

X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train random forest classifier on all train and validation
# test data: evaluate

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)
print("loss= ",score,"%")