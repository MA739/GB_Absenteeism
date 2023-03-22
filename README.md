# GB_Absenteeism

You need to execute the scripts in this order. 
1. preprocessingScript.py
2. labelDiscretization.py
3. gradientBoostingScript.py

Original dataset provided by Tony Priyanka at https://www.kaggle.com/datasets/tonypriyanka2913/employee-absenteeism. 

Preprocessing Script Manual
	This script is used to preprocess the original dataset[Absenteeism_at_work_Project.csv]. It iterates through the data features and fills in blank values with zeroes or the average of the feature. The user can dictate this by commenting out line 126 or 127.
	It uses the file "Absenteeism_at_work_Project.csv". Before you execute this script, you must set the dataPath variable to the complete filepath of the "Absenteeism_at_work_Project.csv" file. You must also set the "fileName" variables, located on line 81 & 112, to a path where you want the output file to be saved.

Label Discretization Manual
	This script converts the "abenteeism hours" data from numeric to string based on a pre-set value range. 0 hours -> A. 1-15 hours -> B. 16-120 hours -> C. This allows for the classifcation algorithms to perform efficiently.
	Before running this script, you must set "input" variable to the location of the preprocessed data file that you created after running the preprocessingScript. Set the "output" variable to the complete path of where you want the new file to be created.

Gradient Boosting Script Manual
	This file classifies the abenteeism hours problem using the Gradient Boosting Classifier, K-Best feature selection, and MRMR feature selection. 
	Before running, set the "input" variable to complete path to the file you created after running the Label Discretization script. 
	Next, set the hyperparameters(learning rate, max depth) to values of your choosing. When deciding how many features to select, make sure that numFeatures and mf(max Features) have the same value. 
	Finally, run the script and the program will execute. After execution, it will print out the accuracy results of each method.
