from lazypredict.Supervised import LazyClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

data=pd.read_csv("diabetes.csv")
data.columns
data.describe()
data.info()#no NaN

sns.scatterplot(data=data,x='Age',y='Outcome')
sns.scatterplot(data=data,x='Insulin',y='Age')# to much 0 for Insulin replace them with mean
sns.scatterplot(data=data,x='Glucose',y='Age')# not to much 0 for Glucose replace them with mean
sns.scatterplot(data=data,x='BloodPressure',y='Age')#to much 0 for BloodPressure some of dataset must be dead
sns.scatterplot(data=data,x='SkinThickness',y='Age')#to much 0
sns.scatterplot(data=data,x='BMI',y='Age')#zeros are outlier
sns.scatterplot(data=data,x='DiabetesPedigreeFunction',y='Age')


data["Insulin"].replace(0,data["Insulin"].median(),inplace=True)
data["BloodPressure"].replace(0,data["BloodPressure"].median(),inplace=True)
data["SkinThickness"].replace(0,data["SkinThickness"].median(),inplace=True)
data["BMI"].replace(0,data["BMI"].median(),inplace=True)
data.head()


sns.heatmap(data.corr())
sns.pairplot(data=data,hue='Outcome')


y=data["Outcome"]
x=data.drop("Outcome",axis=1)
x_columns=x.columns

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_x=scaler.fit_transform(x)
x=pd.DataFrame(scaled_x,columns=x_columns)
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25)


lcf=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
model,pred=lcf.fit(x_train,x_test,y_train,y_test)







x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25)

lcf=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
model,pred=lcf.fit(x_train,x_test,y_train,y_test)
print(model)#LGBMClassifier    RandomForestClassifier    ExtraTreesClassifier    CalibratedClassifierCV   LinearDiscriminantAnalysis    RidgeClassifierCV



###

import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


###testing for LGBMClassifier
"""
LGC=lgb.LGBMClassifier()
LGC.fit(x_train,y_train)
y_pred=LGC.predict(x_test)

accuracy=accuracy_score(y_pred, y_test)

print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

#overfitting control
y_train_predict=LGC.predict(x_train)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_train_predict)))



param_grid = {'learning_rate': [0.005,0.01], 'n_estimators' : [2,4,8,16],  'num_leaves': [8,16,32,64,128],'boosting_type' : ['dart'], 'objective' : ['binary'], 'max_bin':[255, 1020],
              'random_state': [500],
              'colsample_bytree' : [0.64, 0.65, 0.66],
              'subsample' : [0.55,0.7],
              'reg_alpha' : [1,1.2],
              'reg_lambda': [0.8, 1.0, 1.2]
              }
lgc_grid=GridSearchCV(lgb.LGBMClassifier(),param_grid,cv=4,n_jobs=-1)
lgc_grid.fit(x_train,y_train)
print(lgc_grid.best_params_)
print(lgc_grid.best_score_)
print(lgc_grid.best_estimator_)
lgb.LGBMClassifier()

"""

lgb=lgb.LGBMClassifier(boosting_type='dart',colsample_bytree=0.64,learning_rate=0.005,max_bin=255, n_estimators=2, num_leaves=8, objective='binary',random_state=500, reg_alpha=1, reg_lambda=0.8, subsample=0.55)
lgb.fit(x_train,y_train)
y_pred=lgb.predict(x_test)
print(classification_report(y_test,y_pred))




#########RandomForestClassifier
print("stop its hammer time ")
from sklearn.ensemble  import RandomForestClassifier
param_grid={'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
rfc_grid=GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=5)
rfc_grid.fit(x_train,y_train)

print(rfc_grid.best_score_)
print(rfc_grid.best_estimator_)
print(rfc_grid.best_params_)

rfc=RandomForestClassifier(rfc_grid.best_params_)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

print(classification_report(y_test,y_pred))

#####ExtraTreesClassifier

print("stop its hammer time ")
from sklearn.ensemble import ExtraTreesClassifier

param_grid={'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
etc_grid=GridSearchCV(ExtraTreesClassifier(),param_grid=param_grid,cv=5)

etc_grid.fit(x_train,y_train)


print(etc_grid.best_score_)
print(etc_grid.best_estimator_)
print(etc_grid.best_params_)
etc=ExtraTreesClassifier(etc_grid.best_params_)
etc.fit(x_train,y_train)
y_pred=etc.predict(x_test)
print(classification_report(y_test,y_pred))

print("not fully finished code its ok ")







