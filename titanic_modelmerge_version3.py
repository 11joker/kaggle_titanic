# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:51:39 2017

@author: n
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import bagging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 



warnings.filterwarnings("ignore")
plt.style.use('ggplot')

train = pd.read_csv("D:/data_file/titanic/train.csv")
test = pd.read_csv("D:/data_file/titanic/test.csv")

missing_val_df = pd.DataFrame(index=["Total", "Unique Cabin", "Missing Cabin"])
for name, df in zip(("Training data", "Test data"), (train, test)):
    total = df.shape[0]
    unique_cabin = len(df["Cabin"].unique())
    missing_cabin = df["Cabin"].isnull().sum()
    missing_val_df[name] = [total, unique_cabin, missing_cabin]

train.drop("PassengerId", axis=1, inplace=True)
for df in train, test:
    df.drop("Cabin", axis=1, inplace=True)
    

for df in train, test:
    df["Embarked"].fillna("S", inplace=True)
    for feature in "Age", "Fare":
        df[feature].fillna(train[feature].mean(), inplace=True)

for df in train, test:
    df.drop("Ticket", axis=1, inplace=True)
    
for df in train, test:
    df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q":2})
    df["Sex"] = df["Sex"].map({"female":0, "male":1})
    
for df in train, test:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

for df in train, test:
    titles = list()
    for row in df["Name"]:
        surname, title, name = re.split(r"[,.]", row, maxsplit=2)
        titles.append(title.strip())
    df["Title"] = titles
    df.drop("Name", axis=1, inplace=True)

for df in train, test:
    for key, value in zip(("Mr", "Mrs", "Miss", "Master", "Dr", "Rev"),
                          np.arange(6)):
        df.loc[df["Title"] == key, "Title"] = value
    df.loc[df["Title"] == "Ms", "Title"] = 1
    for title in "Major", "Col", "Capt":
        df.loc[df["Title"] == title, "Title"] = 6
    for title in "Mlle", "Mme":
        df.loc[df["Title"] == title, "Title"] = 7
    for title in "Don", "Sir":
        df.loc[df["Title"] == title, "Title"] = 8
    for title in "Lady", "the Countess", "Jonkheer":
        df.loc[df["Title"] == title, "Title"] = 9
test["Title"][414] = 0
    
nominal_features = ["Pclass", "Sex", "Embarked", "FamilySize", "Title"]
for df in train, test:
    for nominal in nominal_features:
        df[nominal] = df[nominal].astype(dtype="category")
        
feature=["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "FamilySize", "Title"]
scaler=StandardScaler().fit_transform(train[feature].values)
train1=pd.DataFrame(scaler,columns=feature,index=train.index)
train1['Survived']=train['Survived']
train=train1
del train1

tp=test["PassengerId"]
feature=["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "FamilySize", "Title"]
scaler=StandardScaler().fit_transform(test[feature].values)
test1=pd.DataFrame(scaler,columns=feature,index=test.index)
test=test1
del test1       
  
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "FamilySize", "Title"]

predictors1 = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]

NFOLDS=5
SEED=0
kf = KFold(n_splits=NFOLDS, random_state=SEED)
ntrain=len(train)
ntest=len(test)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # split data in NFOLDS training vs testing samples
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # select train and test sample
        x_tr = x_train.iloc[train_index].copy()
        y_tr = y_train.iloc[train_index].copy()
        x_te = x_train.iloc[test_index].copy()
        
        # train classifier on training sample
        clf.fit(x_tr, y_tr)
        
        # predict classifier for testing sample
        oof_train[test_index] = clf.predict(x_te)
        # predict classifier for original test sample
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # take the median of all NFOLD test sample predictions
    # (changed from mean to preserve binary classification)
    oof_test[:] = np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

svc=svm.SVC(C=0.6, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
cross_val_score(svc , train[predictors] , train['Survived'],cv=5)

kn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
cross_val_score(kn , train[predictors] , train['Survived'],cv=5)


dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=7,
            min_samples_split=25, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
cross_val_score(dt , train[predictors] , train['Survived'],cv=5)

svc = svm.SVC(C=0.6, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
bag=bagging.BaggingClassifier(base_estimator=svc,
         bootstrap=True, bootstrap_features=False, max_features=5,
         max_samples=0.7, n_estimators=60, n_jobs=1, oob_score=False,
         random_state=None, verbose=0, warm_start=False)
cross_val_score(bag , train[predictors] , train['Survived'],cv=5)

ada = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.5, n_estimators=18, random_state=None)
cross_val_score(ada , train[predictors] , train['Survived'],cv=5)


forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=4,
            min_samples_split=14, min_weight_fraction_leaf=0.0,
            n_estimators=30, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
cross_val_score(forest , train[predictors] , train['Survived'],cv=5)

xg=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.82,
       gamma=0.91, learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=0.008, missing=None, n_estimators=119, nthread=-1,
       objective='binary:logistic', reg_alpha=0.06, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.87)
cross_val_score(xg, train[predictors] , train['Survived'],cv=5)

svc_oof_train, svc_oof_test = get_oof(svc,train[predictors],train['Survived'],test[predictors])
kn_oof_train, kn_oof_test = get_oof(kn,train[predictors],train['Survived'],test[predictors])
forest_oof_train, forest_oof_test = get_oof(forest,train[predictors1],train['Survived'],test[predictors1])
ada_oof_train, ada_oof_test = get_oof(ada,train[predictors],train['Survived'],test[predictors])
bag_oof_train, bag_oof_test = get_oof(bag,train[predictors],train['Survived'],test[predictors])
dt_oof_train, dt_oof_test = get_oof(dt,train[predictors1],train['Survived'],test[predictors1])
xg_oof_train,xg_oof_test=get_oof(xg,train[predictors],train['Survived'],test[predictors])
#0.86
model_merge_table=pd.DataFrame({'svc':svc_oof_train.ravel(),'kn':kn_oof_train.ravel(),'forest':forest_oof_train.ravel(),'ada':ada_oof_train.ravel(),\
              'bag':bag_oof_train.ravel(),'dt':dt_oof_train.ravel(),'xg':xg_oof_train.ravel()})

plt.figure(figsize=(12,10))
foo = sns.heatmap(model_merge_table.corr(), vmax=1.0, square=True, annot=True)

train_df=model_merge_table
train_df.drop('kn',axis=1,inplace=True)
X_train, X_test, y_train, y_test=train_test_split(train_df,train['Survived'])

#0.85
model=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.82,
       gamma=0.91, learning_rate=0.1, max_delta_step=0, max_depth=2,
       min_child_weight=0.008, missing=None, n_estimators=1, nthread=-1,
       objective='binary:logistic', reg_alpha=0.06, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.91)
model.fit(X_train,y_train)

DTrain = xgb.DMatrix(X_train, y_train)
x_parameters=model.get_params()
xgb_len=len(xgb.cv(x_parameters, DTrain,num_boost_round=400,early_stopping_rounds=50))
model.set_params(n_estimators=xgb_len)

pa={ 'max_depth':range(1,6)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

pa={'subsample':(0.91,0.92,0.89,0.88,0.87,0.9)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

pa={ 'min_child_weight':(0.008,0.009,0.006,0.007)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

pa={ 'gamma':(0.91,0.92,0.9)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_      
model        

pa={'colsample_bytree':(0.82,0.83,0.81)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

pa={'reg_alpha':(0.06,0.07,0.05,0.08,0.09)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

pa={'learning_rate':(0.1,0.12,0.11,0.09,0.13,0.08)}
model=GridSearchCV(model,pa)
model.fit(X_train,y_train)
model=model.best_estimator_
model

model.fit(X_train, y_train)
print("model: {0:.2}".format(model.score(X_test, y_test)))
    
test_df=pd.DataFrame({'svc':svc_oof_test.ravel(),'forest':forest_oof_test.ravel(),'ada':ada_oof_test.ravel(),\
              'bag':bag_oof_test.ravel(),'dt':dt_oof_test.ravel(),'xg':xg_oof_test.ravel()}) 
    
sub = pd.read_csv('D:/data_file/titanic/gender_submission.csv', low_memory=False)
pred_y=model.predict(test_df)    
sub['Survived']=pred_y
sub.to_csv('D:/data_file/kaggle_result/titanic.csv',index=False)   
    
    
    