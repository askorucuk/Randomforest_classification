# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:21:37 2021

@author: askor
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

wine2 = load_wine();

print('Classes', wine2.target_names,"\n");
print('Features', wine2.feature_names,"\n");

X = wine2.data
y = wine2.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y);

clf2=RandomForestClassifier(n_estimators=100)


clf2.fit(X_train,y_train)

y_pred=clf2.predict(X_test)

result_data2 = clf2.predict(X_test);

print("Result : ",result_data2 ,"\n");

print("\nAccuracy score = " + str(accuracy_score(result_data2, y_test))+"\n");

feature_imp = pd.Series(clf2.feature_importances_,index=wine2.feature_names).sort_values(ascending=False)
print(feature_imp);

cm2 = confusion_matrix(y_test, result_data2);
print("\nConfusion Matrix : ");
print(cm2,"\n");


print(cross_val_score(clf2,wine2.data,wine2.target,cv=10));

n = float(input("Please enter the number of features : "));
test_arr1 = input("Please enter the your test data(separate your test data with commas) : ");
test_list1 = list(map(float,test_arr1.split(',')));
n = float(input("Please enter the number of features : "));
test_arr2 = input("Please enter the your second test data(separate your test data with commas) : ");
test_list2 = list(map(float,test_arr2.split(',')));
main_test_list = (test_list1,test_list2);

input_test_result = clf2.predict(main_test_list);
print("\nUser Input Test Result :",input_test_result);










