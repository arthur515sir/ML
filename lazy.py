from lazypredict.Supervised import LazyClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv("diabetes.csv")
x=data.drop("Outcome",axis=1)

y=data["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25)

lcf=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
model,pred=lcf.fit(x_train,x_test,y_train,y_test)
print(model)


