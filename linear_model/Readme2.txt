Input for q2_run.py are a train file and a test file. 
This training dataset is taken from UCI repository (http://archive.ics.uci.edu/ml/index.php)
   
To run the script type:

	python q2_run.py q2_train.csv q2_test.csv

The output is the:
- The prediction for each instance in the test set.
- value of B0 and B1 for logistic regression model  

Train set has 567 instances and test set currently has  10 instances. The format of this test set should be used
when you create your own set of test instances. 

The file has 8 attributes. The name and range of each attribute is given below
Attributes:
	Pregnancies: (0-17)
	Glucose: (0-199)
	BloodPressure (0-122)
	SkinThickness (0-99)
	Insulin (0-846)
	BMI (0-67.1)
	DiabetesPedigreeFunction (0.078-2.42)
	Age (21-81)

Column one is the outcome of each instance which must either be 0 or 1
