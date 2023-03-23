from sklearn.ensemble import GradientBoostingClassifier
#import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings("ignore")

def featSelectMRMR(in_table, lrr, mdd, mff, nff):
    #select the most relevant columns, then predict based on those
    print("using MRMR")
    #input = pd.read_csv('C://Users//Dr4gonborn//Desktop//DM_Project//PAH[ZeroedAbsHours].csv')
    df = in_table.copy()
    targetColumn = df["Absenteeism time in hours"]
    df.drop("Absenteeism time in hours", axis=1, inplace=True)
    #print(targetColumn)
    #print(df['Absenteeism time in hours'])
    X = df
    y = pd.Series(targetColumn)
    #selecting for the top 10 most relevant columns
    selected_features = mrmr_classif(X=X, y=y, K=nff)
    print("Top 10 columns for prediction")
    print(selected_features)

    #the top 10 columns selected were:
    #['Disciplinary failure', 'Service time', 'Day of the week', 'Work load Average/day ',
    #'Reason for absence', 'Son', 'Height', 'Hit target', 'Seasons', 'Transportation expense']

    #using this, we can run a prediction using these columns only
    #first, we drop all unrelated columns
    #labeledTable = pd.read_csv('C://Users//Dr4gonborn//Desktop//DM_Project//PAH[ZeroAB_Labeled].csv')
    mlTargetValue = targetColumn
    X_train, X_test, y_train, y_test = train_test_split(df.filter(selected_features), mlTargetValue, test_size=0.2)

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Define Gradient Boosting Classifier with hyperparameters
    gbc = GradientBoostingClassifier(criterion='friedman_mse', n_estimators=500, learning_rate=lrr,
                                     loss='deviance', random_state=100, max_features=mff, max_depth=mdd)
    # Fit train data to GBC
    gbc.fit(X_train_transformed, y_train)
    # Confusion matrix will give number of correct and incorrect classifications
    print(confusion_matrix(y_test, gbc.predict(X_test_transformed)))
    # Accuracy of model
    print("MRMR Feature Selection: GBC accuracy is %2.2f" % accuracy_score(
        y_test, gbc.predict(X_test_transformed)))


def featSelectKBest(inputData, lrr, mdd, mff ,nff):
    print("using SelectKBest")
    id_norm = inputData.copy()
    label = id_norm['Absenteeism time in hours']
    id_norm.drop("Absenteeism time in hours", axis=1, inplace=True)
    #print("Original dataframe shape: ")
    #print(id_norm.shape)
    X_New = SelectKBest(k=nff, score_func=chi2).fit_transform(id_norm, label)
    #proves that the best categories were chosen. Need to figure out which columns were left over
    #print("KBest columns chosen ")
    #print(X_New.shape)

    #print("Inverting")
    #fixedX = X_New.transform()
    #X_New.inverse_transform(X)
    #print(X_New)
    #print("KBest complete. Ready to do prediction")
    X_train, X_test, y_train, y_test = train_test_split(X_New,
                                                        label, test_size=0.2)

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Define Gradient Boosting Classifier with hyperparameters
    gbc = GradientBoostingClassifier(criterion='friedman_mse', n_estimators=500, learning_rate=lrr,
                                     loss='deviance', random_state=100, max_features=mff, max_depth=mdd)
    # Fit train data to GBC
    gbc.fit(X_train_transformed, y_train)
    # Confusion matrix will give number of correct and incorrect classifications
    print(confusion_matrix(y_test, gbc.predict(X_test_transformed)))
    # Accuracy of model
    print("KBEST Feature selection: GBC accuracy is %2.2f" % accuracy_score(
        y_test, gbc.predict(X_test_transformed)))

def noFeatSelect(input_Table, lr, md):
    X_train, X_test, y_train, y_test = train_test_split(input_Table.drop('Absenteeism time in hours', axis=1),
                                                        input_Table['Absenteeism time in hours'], test_size=0.2)

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Define Gradient Boosting Classifier with hyperparameters. Uses all parameters...
    gbc = GradientBoostingClassifier(criterion='friedman_mse', n_estimators=500, learning_rate=lr,
                                     loss='deviance', random_state=100, max_depth=md)
    # Fit train data to GBC
    gbc.fit(X_train_transformed, y_train)

    # Confusion matrix will give number of correct and incorrect classifications
    print(confusion_matrix(y_test, gbc.predict(X_test_transformed)))
    # Accuracy of model
    print("NO Feature Selection: GBC accuracy is %2.2f" % accuracy_score(
        y_test, gbc.predict(X_test_transformed)))

#Conducts experiment using the average substitution table
def avgSubExperiment(lr, md, mf, nF):
    input = pd.read_csv('C:Path//PAH[AvgAB_Labeled].csv')
    input.head()
    # ID is only for identification. It is not relevant to predict outcome
    input.drop("ID", axis=1, inplace=True)
    # drop Absenteeism time in hours

    # Split dataset into test and train data
    featSelectKBest(input, lr, md, mf, nF)
    # uses the raw absentee numbers instead of labels
    featSelectMRMR(input, lr, md, mf, nF)

    noFeatSelect(input, lr, md)

#Conducts experiment using the zero substitution table
def zeroSubExperiment(lr, md, mf, nF):
    input = pd.read_csv('C:Path//PAH[ZeroAB_Labeled].csv')
    input.head()
    # ID is only for identification. It is not relevant to predict outcome
    input.drop("ID", axis=1, inplace=True)
    # drop Absenteeism time in hours

    # Split dataset into test and train data
    featSelectKBest(input, lr, md, mf, nF)
    # uses the raw absentee numbers instead of labels
    featSelectMRMR(input, lr, md, mf, nF)

    noFeatSelect(input, lr, md)

#In the current state, this script classifies before feature selection
if __name__ == "__main__":
    # variables for filling in hyperparameters

    # Learning Rate
    # Utilized Learning rates of .1 or .5
    learning_rate = 0.1

    # Max Depth
    # Utilized Max Depth of 3 or 5
    max_depth = 5

    # Max Features -> Must be equivalent to numFeatures
    # Utilized Max Features of 5 or 10
    max_features = 10

    # top number of features used for classification.
    # Utilized value of 5 or 10
    numFeatures = 10

    print("Zero Substitution")
    zeroSubExperiment(learning_rate,max_depth, max_features, numFeatures)
    print("Average Substitution")
    avgSubExperiment(learning_rate,max_depth, max_features, numFeatures)
