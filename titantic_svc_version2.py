# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:37:05 2017

@author: n
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import svm 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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
  
from sklearn.model_selection import train_test_split

predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train["Survived"])

svc = svm.SVC(C=0.9, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

pa={ 'C':[0.4,0.5,0.3]}
svc=GridSearchCV(svc,pa)
svc.fit(X_train,y_train)
svc=svc.best_estimator_
svc

svc.fit(X_train, y_train)
print("SVC: {0:.2}".format(svc.score(X_test, y_test)))

predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
clf = svm.SVC(C=0.8, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(train[predictors], train["Survived"])
prediction = clf.predict(test[predictors])

submission = pd.DataFrame({"PassengerId": tp, "Survived": prediction})
submission.to_csv("D:/data_file/kaggle_result/titanic.csv", index=False)
