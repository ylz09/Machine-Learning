#!/usr/bin/python

"""
Script for ML class
compare linear regression and logistic regression
Author: Pakeeza Akram and Yuqi Kong
Date: Feb 20, 2018
"""
import sys
import numpy as np
from sklearn import linear_model as lin_log
#import matplotlib.pyplot as plt


#load the training and the test set
Train_set = sys.argv[1]     
data = np.loadtxt(Train_set, delimiter=',')  # load training data
x_train = data[:, 1]
y_train = data[:, 0]
x_train = x_train[:, np.newaxis]
Test_set = sys.argv[2]    
data = np.loadtxt(Test_set, delimiter=',')  # load test data
x_test = data[:, 1]
y_test = data[:, 0]
x_test = x_test[:, np.newaxis]



def model(x):
    return 1 / (1 + np.exp(-x))



# run the classifier
# run the classifier
#logistic regression
logistic = lin_log.LogisticRegression(C=1e5)
logistic.fit(x_train, y_train)
#linear regression
linear = lin_log.LinearRegression()
linear.fit(x_train, y_train)
#perceptron
perceptron = lin_log.Perceptron(max_iter = 500)
perceptron.fit(x_train, y_train)

print
print "For Linear Regression: intercept: {:.6f}, coef: {:.6f}".format(linear.intercept_, linear.coef_[0])
print "For Logistic Regression: :B0 {:.6f}, B1: {:.6f}".format(logistic.intercept_[0], logistic.coef_[0][0])
print "For perceptron model: w0: {:.6f}, w1: {:.6f}".format(perceptron.intercept_[0], perceptron.coef_[0][0])



#test the classifier
log_y=logistic.predict(x_test)
lin_y=linear.predict(x_test)
per_y=perceptron.predict(x_test)


#Prediction results saved in a file call Prediction_results
fw=open("Prediction_results.csv","w")
fw.write("X-value" + "," + "Logistic Prediction" +","+"Linear Regression Prediction"+","+"Perceptron Regression"+"\n")
for i in range(len(log_y)):
	fw.write(str(x_test[i]) + "," + str(log_y[i])+","+str(lin_y[i])+","+str(per_y[i])+"\n")
fw.close()

# def model(x):
    # return 1 / (1 + np.exp(-x))

# #Plot of logistic and linear regression
# plt.figure()
# loss = model(x_test* logistic.coef_ + logistic.intercept_).ravel()
# plt.plot(x_test, loss, color='red', linewidth=1)
# plt.plot(x_test, linear.coef_ * x_test + linear.intercept_, linewidth=1)
# plt.plot(x_test, perceptron.predict(x_test), linewidth=1)
# plt.scatter(x_test,log_y, color='red')
# plt.scatter(x_test,lin_y, color='blue')
# plt.scatter(x_test,per_y, color='green')
# plt.ylabel('y')
# plt.xlabel('X')
# plt.grid(True)
# plt.legend(('Logistic Regression Model', 'Linear Regression Model', 'Perceptron Model'),
           # loc="lower right", fontsize='small')
# plt.show()
