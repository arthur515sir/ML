import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib as plt
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/juliencohensolal/BankMarketing/master/rawData/bank-additional-full.csv",sep=';')

#preproceesing
y=df.iloc[:,-1]
x=df.iloc[:,:-1]


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn import preprocessing


x=pd.get_dummies(x,columns=["job","marital","education","default","housing","contact","month","day_of_week","poutcome","marital","marital","marital","loan"],drop_first=True)
x_columns=x.columns
min_max=preprocessing.MinMaxScaler()
x=min_max.fit_transform(x)
x=pd.DataFrame(x,columns=x_columns)



def corr_func(df):
    for i in df:
        print(i)


y=y.replace("no",0)
y=y.replace("yes",1)


df_new=pd.concat([x,y],axis=1)
cor_df=df_new.corr()

print(corr_func(cor_df[:]["y"]))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,stratify=y)



#Logistic regression
print("stop its hammer time ")
param_grid = {'C': [0.1,0.5,1,10], 'max_iter' : [500, 750], 'tol':[0.00001,0.0001, 0.001]}
lr_grid=GridSearchCV(LogisticRegression(solver='liblinear'),param_grid,cv=5)
lr_grid.fit(x_train,y_train)


print(lr_grid.best_score_)
print(lr_grid.best_estimator_)
print(lr_grid.best_params_)

LR=LogisticRegression(C=10,max_iter=500,solver='liblinear',tol=0.001)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
print(classification_report(y_test,y_pred))
#KNeighborsClassifier
print("stop its hammer time ")

param_grid = {'n_neighbors': list(range(2,5,1)), 'algorithm' : ["auto", "ball_tree","kd_tree","brute"]}
KNC_grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
KNC_grid.fit(x_train,y_train)

print(KNC_grid.best_score_)
print(KNC_grid.best_estimator_)
print(KNC_grid.best_params_)


KNC=KNeighborsClassifier(n_neighbors=4,algorithm='auto')
KNC.fit(x_train,y_train)
y_pred=KNC.predict(x_test)

print(classification_report(y_test,y_pred))

### SVC classifier
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
#
print("cant touch this ")
