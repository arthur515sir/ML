# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:50:31 2022

@author: MSI
"""
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
df=pd.read_csv("C:\\Users\\Guven Nazlican\\Desktop\\ML\\creditcard.csv")

RS=RobustScaler()
df["Amount"]=RS.fit_transform(df[["Amount"]])

fraud_df_train = df.loc[df['Class'] == 1]
non_fraud_df_train = df.loc[df['Class'] == 0][:round(3*len(fraud_df_train))]


train=pd.concat([fraud_df_train,non_fraud_df_train])
train=train.sample(frac=1)

y=train["Class"]
x=train.iloc[:,1:30]


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25)

#Logistic regression
print("stop its hammer time ")
param_grid = {'C': [0.1,0.5,1,10], 'max_iter' : [500, 750], 'tol':[0.00001,0.0001, 0.001]}
lr_grid=GridSearchCV(LogisticRegression(solver='liblinear'),param_grid,cv=5)
lr_grid.fit(x_train,y_train)

print(lr_grid.best_score_)
print(lr_grid.best_estimator_)
print(lr_grid.best_params_)
LR=LogisticRegression(C=0.5,max_iter=500,solver='liblinear',tol=1e-05)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print(classification_report(y_test,y_pred))
### KNeighborsClassifier
print("stop its hammer time ")
param_grid = {'n_neighbors': list(range(2,5,1)), 'algorithm' : ["auto", "ball_tree","kd_tree","brute"]}
KNC_grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
KNC_grid.fit(x_train,y_train)

KNC=KNeighborsClassifier(n_neighbors=3,algorithm='auto')
KNC.fit(x_train,y_train)
y_pred=KNC.predict(x_test)
print(classification_report(y_test,y_pred))
### SVM classifier
print("stop its hammer time")
param_grid={'C':[0.5,0.7,0.9,1,],'kernel':['rbf', 'poly', 'sigmoid', 'linear']}

SVC_grid=GridSearchCV(SVC(),param_grid,cv=5)
SVC_grid.fit(x_train,y_train)
print(SVC_grid.best_params_)
print(SVC_grid.best_score_)
print(SVC_grid.best_estimator_)
SVC_Model=SVC(C=0.7,kernel='linear')
SVC_Model.fit(x_train,y_train)
y_pred=SVC_Model.predict(x_test)
print(classification_report(y_test,y_pred))
#######
print("cant touch this ")

