# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:46:03 2018

@author: fzhan
"""
import pandas as pd
train=pd.read_csv('train.csv')
train=train.drop('Unnamed: 0',axis=1)
validate=pd.read_csv('validate.csv')
validate=validate.drop('Unnamed: 0',axis=1)
train_y=train.loan_status
train_x=train.drop('loan_status',axis=1)
validate_x=validate.drop('loan_status',axis=1)
validate_y=validate.loan_status

from sklearn.preprocessing import StandardScaler
train_x1=StandardScaler().fit_transform(train_x)
validate_x1=StandardScaler().fit_transform(validate_x)

from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score
names=['lr','sgdc','qda','rfc','abc']
classifiers = [LogisticRegression(),
               SGDClassifier(),
               #SVC(kernel="linear", C=0.025),
               #SVC(gamma=2, C=1),
               #QuadraticDiscriminantAnalysis(),
               RandomForestClassifier(max_depth=5, n_estimators=20, max_features=8),
               AdaBoostClassifier()]
lis=list(zip(names,classifiers))
i=0
while i<len(lis):
    clf=lis[i][1]
    print(cross_val_score(clf,train_x1,train_y,cv=3,scoring='accuracy'))
    pred_y=cross_val_predict(clf,train_x1,train_y,cv=3)
    print(confusion_matrix(train_y,pred_y))
    print(roc_auc_score(train_y,pred_y))
    i+=1
    

    
import pickle
s = pickle.dumps(clf)
clf = pickle.loads(s)

from sklearn.ensemble import GradientBoostingClassifier
clf2=GradientBoostingClassifier()
print(cross_val_score(clf2,train_x1,train_y,cv=3,scoring='accuracy'))
pred_y2=cross_val_predict(clf2,train_x1,train_y,cv=3)
print(confusion_matrix(train_y,pred_y2))
print(roc_auc_score(train_y,pred_y2))



from xgboost.sklearn import XGBClassifier

clf3= XGBClassifier( learning_rate=0.1, n_estimators=50, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=5,seed=27)
print(cross_val_score(clf3,train_x1,train_y,cv=3,scoring='roc_auc'))
pred_y3=cross_val_predict(clf3,train_x1,train_y,cv=3)
print(confusion_matrix(train_y,pred_y3))
print(roc_auc_score(train_y,pred_y3))
print(precision_score(train_y,pred_y3))


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
param_grid=[{'learning_rate':[0.03,0.1,0.2]},{'n_estimators':[50,100,150],'max_depth':[5,10],
             'subsample':[0.5,0.8,1]}]
#define your own measure
score_measure=make_scorer(roc_auc_score)
grid_search=GridSearchCV(clf3,param_grid,cv=3,scoring=score_measure)
grid_search.fit(train_x1,train_y)
grid_search.best_params_
cvres=grid_search.cv_results_
grid_search.best_score_
grid_search.best_estimator_


for score,params in zip(cvres['split2_test_score'],cvres['params']):
    print(score,params)
    
grid_search.predict_proba(validate_x1)
pred_y=grid_search.predict(validate_x1)
print(confusion_matrix(validate_y,pred_y))
print(recall_score(validate_y,pred_y))
print(roc_auc_score(validate_y,pred_y))
print(accuracy_score(validate_y,pred_y))




    


    
