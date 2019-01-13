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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
    
non_empty_embarked = train["Embarked"].dropna()
unique_values, value_counts = non_empty_embarked.unique(),\
  non_empty_embarked.value_counts()
X = np.arange(len(unique_values))
colors = ["brown", "grey", "purple"] 

plt.bar(left=X,
        height=value_counts,
        color=colors,
        tick_label=unique_values) 
plt.xlabel("Port of Embarkation")
plt.ylabel("Amount of embarked")
plt.title("Bar plot of embarked in Southampton, Queenstown, Cherbourg")

survived = train[train["Survived"] == 1]["Age"].dropna()
perished = train[train["Survived"] == 0]["Age"].dropna()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 6)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='green', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax2.hist(perished, facecolor='brown', alpha=0.75)
ax2.set(title="Dead", xlabel="Age", ylabel="Amount")

'''survived = train[train["Survived"] == 1]["Age"].dropna()
perished = train[train["Survived"] == 0]["Age"].dropna()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 6)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='green', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax1.hist(perished, facecolor='brown', alpha=0.75)
ax1.set(title="Dead", xlabel="Age", ylabel="Amount")'''

survived = train[train["Survived"] == 1]["Fare"].dropna()
perished = train[train["Survived"] == 0]["Fare"].dropna()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 8)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='darkgreen', alpha=0.75)
ax1.set(title="Survived", xlabel="Fare", ylabel="Amount")
ax2.hist(perished, facecolor='darkred', alpha=0.75)
ax2.set(title="Dead", xlabel="Fare", ylabel="Amount")

'''
survived = train[train["Survived"] == 1]["Fare"].dropna()
perished = train[train["Survived"] == 0]["Fare"].dropna()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 8)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='darkgreen', alpha=0.75)
ax1.set(title="Survived", xlabel="Fare", ylabel="Amount")
ax1.hist(perished, facecolor='darkred', alpha=0.75)
ax1.set(title="Dead", xlabel="Fare", ylabel="Amount")
'''

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

title = train["Title"]
unique_values, value_counts = title.unique(), title.value_counts()
X = np.arange(len(unique_values))

fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
ax.bar(left=X, height=value_counts, width=0.5, tick_label=unique_values)
ax.set_xlabel("Title")
ax.set_ylabel("Count")
ax.set_title("Passenger titles")
ax.grid(color='g', linestyle='--', linewidth=0.5)

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
        
from sklearn.model_selection import train_test_split

predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train["Survived"])

forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=4,
            min_samples_split=14, min_weight_fraction_leaf=0.0,
            n_estimators=30, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

pa={ 'n_estimators':list(range(2,200,4))}
forest=GridSearchCV(forest,pa)
forest.fit(X_train,y_train)
forest=forest.best_estimator_
forest

pa={ 'max_depth':list(range(2,7,1))}
forest=GridSearchCV(forest,pa)
forest.fit(X_train,y_train)
forest=forest.best_estimator_
forest

pa={ 'min_samples_split':list(range(4,50,2))}
forest=GridSearchCV(forest,pa)
forest.fit(X_train,y_train)
forest=forest.best_estimator_
forest

pa={ 'min_samples_leaf':list(range(4,60,2))}
forest=GridSearchCV(forest,pa)
forest.fit(X_train,y_train)
forest=forest.best_estimator_
forest

forest.fit(X_train, y_train)
print("Random Forest score: {0:.2}".format(forest.score(X_test, y_test)))


predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(train[predictors], train["Survived"])
prediction = clf.predict(test[predictors])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("D:/data_file/kaggle_result/titanic.csv", index=False)
