# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:50:31 2022

@author: MSI
"""
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\MSI\\Desktop\\archive\\creditcard.csv")

print(df[:10])
print(df.shape)
print(df["Class"])


pos=[a for a in df["Class"] if a ==1] 
neg=[a for a in df["Class"] if a ==0] 

print(str ((len(pos)/len(df))*100)+"% is fraud  data ")
y=(df.iloc[:,30:31])
x=(df.iloc[:,1:30])
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25)






    
    

