
To run the script type: 
	
	python q1_run.py q1_train.csv q1_test.csv

This script will take input files q1_train.csv as training set, q1_test.csv as test set.
Train set has 82 instances and test set currently has 20 instances. 
Anything above 1.0 is 1 and below -1.0 is 0. Use the format of test set to modify it to test
on instances you will need to create.

The output is the:
- value of coefficient and intercept for linear regression learned model.
- value of B0 and B1 for logistic regression learned model.
- value of w0 and w1 for perceptron learned model.
 
- The program also generates a file named prediction_results.csv file which contains all 3 models' prediction on the test set. 


