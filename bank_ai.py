import pandas as pd
import matplotlib as plt
df=pd.read_csv("https://raw.githubusercontent.com/juliencohensolal/BankMarketing/master/rawData/bank-additional-full.csv",sep=';')


#preproceesing
y=df.iloc[:,-1]
x=df.iloc[:,:-1]

from sklearn import preprocessing


x=pd.get_dummies(x,columns=["job","marital","education","default","housing","contact","month","day_of_week","poutcome","marital","marital","marital","loan"],drop_first=True)
x_columns=x.columns
min_max=preprocessing.MinMaxScaler()
x=min_max.fit_transform(x)
x=pd.DataFrame(x,columns=x_columns)



