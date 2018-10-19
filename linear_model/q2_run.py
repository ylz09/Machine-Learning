#!/usr/bin/python

"""
Script for ML class
logistic regression on Diabetes data-set
Author: Pakeeza Akram
Date: Feb 20, 2018
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as log_reg

#load the training and the test set
Train_set = sys.argv[1]     
data_trn = np.loadtxt(Train_set, delimiter=',', ndmin = 2)  # load train data
Test_set = sys.argv[2]    
data_tst = np.loadtxt(Test_set, delimiter=',', ndmin = 2)  # load test data

x_train = data_trn[:, 1]
y_train = data_trn[:, 0]
x_train = x_train[:, np.newaxis]
x_test = data_tst[:, 1]
y_test = data_tst[:, 0]
x_test = x_test[:, np.newaxis]


# run the classifier
#logistic regression
logistic = log_reg.LogisticRegression(C=1e5)
logistic.fit(x_train, y_train)

#test the classifier
log_y=logistic.predict(x_test)

print
print "Prediction based on test set:"
for index in range(0, log_y.size):
	print "Row {}\tprediction:-> {:.0f}".format(index+1, log_y[index])

print
print "For Logistic Regression: :B0 {:.6f}, B1: {:.6f}".format(logistic.intercept_[0], logistic.coef_[0][0])
