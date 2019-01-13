# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:08:29 2017

@author: n
"""
'''
PassengerId      int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
'''
'''the score test is '''
'''the cross validation is 
array([ 0.86592179,  0.81005587,  0.87640449,  0.80898876,  0.8700565 ])'''

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
import numpy as np

train_df = pd.read_csv('D:/data_file/titanic/train.csv', low_memory=False)
test_df = pd.read_csv('D:/data_file/titanic/test.csv', low_memory=False)
sub = pd.read_csv('D:/data_file/titanic/gender_submission.csv', low_memory=False)
y_train=train_df['Survived']
train_df.drop('Survived',axis=1,inplace=True)

train_df.drop('PassengerId',axis=1,inplace=True)

train_df['Cabin'].fillna(0,inplace=True)
train_df['Cabin'][train_df.Cabin!=0]=1
train_df['Cabin']=train_df['Cabin'].astype('int32')

train_df['Ticket_Lett'] = train_df['Ticket'].apply(lambda x: str(x)[0])
train_df['Ticket_Lett'] = train_df['Ticket_Lett'].apply(lambda x: str(x))
train_df['Ticket_Lett'] = np.where((train_df['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train_df['Ticket_Lett'],
                                    np.where((train_df['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
train_df['Ticket_Len'] = train_df['Ticket'].apply(lambda x: len(x))
del train_df['Ticket']
Ticket_Lett=pd.get_dummies(train_df['Ticket_Lett'],prefix='Ticket_Lett')
train_df=pd.concat([train_df,Ticket_Lett],axis=1)

train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.')
train_df['Initial'].value_counts().to_dict()
train_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',
                         'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',
                                          'Other','Other','Other','Mr','Mr','Other'],inplace=True)
'''lbl=LabelEncoder()
lbl.fit(list(train_df.Initial.unique()))
train_df['Initial']=lbl.transform(train_df['Initial'])'''

Initial=pd.get_dummies(train_df.Initial,prefix='Initial')
train_df=pd.concat([train_df,Initial],axis=1)
train_df.drop('Initial',axis=1,inplace=True)

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
train_df['Family_size']=train_df.SibSp+train_df.Parch
        
train_df['SibSp_num']=0
train_df['SibSp_num'][(train_df.SibSp>0) & (train_df.SibSp<=2)]=1
train_df['SibSp_num'][(train_df.SibSp>2) & (train_df.SibSp<=4)]=2
train_df['SibSp_num'][(train_df.SibSp>5) & (train_df.SibSp<=8)]=3

train_df['Parch_num']=0
train_df['Parch_num'][train_df.Parch==0]=0
train_df['Parch_num'][(train_df.Parch>0) & (train_df.SibSp<=3)]=1
train_df['Parch_num'][(train_df.SibSp>3) & (train_df.SibSp<=8)]=2       

train_df['Embarked'].replace(['C','Q','S'],[0,1,2],inplace=True)


train_df['is_child']=(train_df.Age<=12).copy()

train_df.drop(['Ticket_Lett','Name','Initial_Mr','Initial_Other','is_child'],axis=1,inplace=True)

'''array([ 0.8603352 ,  0.84357542,  0.86516854,  0.80898876,  0.8700565 ])'''
model=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.82,
       gamma=0.91, learning_rate=0.09, max_delta_step=0, max_depth=5,
       min_child_weight=0.008, missing=None, n_estimators=81, nthread=-1,
       objective='binary:logistic', reg_alpha=0.05, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.91)

#SibSp,Parch

model.fit(train_df,y_train)

DTrain = xgb.DMatrix(train_df, y_train)
x_parameters=model.get_params()
xgb_len=len(xgb.cv(x_parameters, DTrain,num_boost_round=400,early_stopping_rounds=50))
model.set_params(n_estimators=xgb_len)

pa={ 'max_depth':range(2,6)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

pa={'subsample':(0.91,0.92,0.89,0.88,0.87,0.9)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

pa={ 'min_child_weight':(0.008,0.009,0.006,0.007)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

pa={ 'gamma':(0.91,0.92,0.9)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_      
model        

pa={'colsample_bytree':(0.82,0.83,0.81)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

pa={'reg_alpha':(0.06,0.07,0.05,0.08,0.09)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

pa={'learning_rate':(0.1,0.12,0.11,0.09,0.13,0.08)}
model=GridSearchCV(model,pa)
model.fit(train_df,y_train)
model=model.best_estimator_
model

cross_val_score(model,train_df,y_train,cv=5)

feature=train_df.columns
arg_list=model.feature_importances_.argsort().tolist()
feature[arg_list][::-1]

test_df.drop('PassengerId',axis=1,inplace=True)

test_df['Cabin'].fillna(0,inplace=True)
test_df['Cabin'][test_df.Cabin!=0]=1
test_df['Cabin']=test_df['Cabin'].astype('int32')

test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.')
test_df['Initial'].value_counts().to_dict()
test_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',
                         'Rev','Capt','Sir','Dona'],['Miss','Miss','Miss','Mr','Mr','Other','Mrs',
                                          'Other','Other','Other','Mr','Mr','Other'],inplace=True)

'''lbl=LabelEncoder()
lbl.fit(list(test_df.Initial.unique()))
test_df['Initial']=lbl.transform(test_df['Initial'])'''

Initial=pd.get_dummies(test_df.Initial,prefix='Initial')
test_df=pd.concat([test_df,Initial],axis=1)

test_df.drop('Initial',axis=1,inplace=True)

test_df['Ticket_Lett'] = test_df['Ticket'].apply(lambda x: str(x)[0])
test_df['Ticket_Lett'] = test_df['Ticket_Lett'].apply(lambda x: str(x))
test_df['Ticket_Lett'] = np.where((test_df['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test_df['Ticket_Lett'],
                                    np.where((test_df['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
test_df['Ticket_Len'] = test_df['Ticket'].apply(lambda x: len(x))
del test_df['Ticket']
Ticket_Lett=pd.get_dummies(test_df['Ticket_Lett'],prefix='Ticket_Lett')
test_df=pd.concat([test_df,Ticket_Lett],axis=1)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Family_size']=test_df.SibSp+test_df.Parch     
       
test_df['SibSp_num']=0
test_df['SibSp_num'][(test_df.SibSp>0) & (test_df.SibSp<=2)]=1
test_df['SibSp_num'][(test_df.SibSp>2) & (test_df.SibSp<=4)]=2
test_df['SibSp_num'][(test_df.SibSp>5) & (test_df.SibSp<=8)]=3

test_df['Parch_num']=0
test_df['Parch_num'][test_df.Parch==0]=0
test_df['Parch_num'][(test_df.Parch>0) & (test_df.SibSp<=3)]=1
test_df['Parch_num'][(test_df.SibSp>3) & (test_df.SibSp<=8)]=2
       
test_df['Embarked'].replace(['C','Q','S'],[0,1,2],inplace=True)

test_df['is_child']=(test_df.Age<=12).copy()

test_df.drop(['Ticket_Lett','Name','Initial_Mr','Initial_Other','is_child'],axis=1,inplace=True)


model.fit(train_df,y_train)
feature=train_df.columns
pred_y=model.predict(test_df[feature])

sub['Survived']=pred_y

sub.to_csv('D:/data_file/kaggle_result/titanic.csv',index=False)