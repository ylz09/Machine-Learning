#!/usr/bin/python

"""
Revised script for ML class
logistic regression on Diabetes data-set
Author: Yuqi Kong
Date: Feb 25, 2018
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

x_train = data_trn[:, 1:9]
y_train = data_trn[:, 0]
#x_train = x_train[:, np.newaxis]
x_test = data_tst[:, 1:9]
y_test = data_tst[:, 0]
#x_test = x_test[:, np.newaxis]


# run the classifier
#logistic regression
logistic = log_reg.LogisticRegression(C=1e5)
logistic.fit(x_train, y_train)

#test the classifier
log_y=logistic.predict(x_test)

print
print "Prediction based on test set:"
for index in range(0, log_y.size):
	print "Instance {}\tprediction:-> {:.0f}".format(index+1, log_y[index])
	
print
print "Model's parameter (Beta):"
print r"Beta 0 (interception): {:.6f}".format(logistic.intercept_[0])
for index in range(0, logistic.coef_.size):
	print r"Beta {} (coefficient): {:.6f}".format(index + 1,logistic.coef_[0][index])
