import csv
import pandas as pd
import math
import random
import numpy as np


def readCSVtoORIGINAL(filePath, head, oDataSet):
    file = open(filePath)
    csvreader = csv.reader(file)
    # next(csvreader)
    head.append(next(csvreader))
    # index = 10
    for line in csvreader:
        oDataSet.append(line)
        # if index > 0:
        #    oDataSet.append(line)
        #    index -= 1
    file.close()


# Fills in any blanks/null values with the column average
def avgSubstitution(table, colNum):
    #   for each column in the table, iterate through the rows until you reach the end and calculate the average.
    #   round the avg up or down, then plug it into the blank spots in the columns
    #print("verifying prints")
    # row, column
    # print(table[0])
    # rowNum = 0
    Sum = 0
    Tally = 0
    Average = 0
    for row in table:
        if row[colNum].strip().isnumeric():
            print(row[colNum])
            Sum += int(row[colNum].strip())
        Tally += 1
    Average = math.floor(Sum / Tally)
    #print("Tally: " + str(Tally))
    #print("Sum: " + str(Sum))
    #print("Avg: " + str(Average))

    # print("Before filling in blank: " + str(table[739][colNum]))
    for row in table:
        if row[colNum].strip().isnumeric() == False:
            row[colNum] = Average
            #print("applied to row: ", row)


# zeroSubstitution
def zeroSubstitution(table, colNum):
    # Fills in blank spots in the 'Absentee Time in Hours' column with 0, assuming that unentered values are equivalent
    # to 0 absentee hours recorded
    print("zeroing blank data fields")
    blankVal = 0
    for row in table:
        if row[colNum].strip().isnumeric() == False:
            row[colNum] = blankVal


# creates and saves a dataset where null values are replaced with 0
def createZeroSubData(dPath):
    header = []

    # contains original data points
    o_data_set = []

    readCSVtoORIGINAL(dPath, header, o_data_set)

    colNumber = 1
    while colNumber <= 20:
        if colNumber == 20:
            zeroSubstitution(o_data_set, colNumber)
        else:
            zeroSubstitution(o_data_set, colNumber)
        colNumber += 1
    print("Preprocessing Done")

    # PAH -> Preprocessed Absentee Hours
    #indicate where you want the output file saved
    filename = 'C:Path//PAH[ZeroAbsHours].csv'
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # 2. create a csvwriter object
        csvwriter.writerow(header[0])  # 4. write the header
        csvwriter.writerows(o_data_set)
    print("Created and saved database where Nulls are converted to 0")


# creates and saves a dataset where null values are replaced with the column average
def createAvgSubData(dPath):
    # contains the column names
    header = []

    # contains original data points
    o_data_set = []

    readCSVtoORIGINAL(dPath, header, o_data_set)

    colNumber = 1
    while colNumber <= 20:
        if colNumber == 20:
            # fills in any blank data entries with 0
            avgSubstitution(o_data_set, colNumber)
        else:
            avgSubstitution(o_data_set, colNumber)
            # ZeroAbsenteeHours(o_data_set, colNumber)
        colNumber += 1
    # print("Preprocessing Done")

    #PAH -> Preprocessed Absentee Hours
    #indicate where you want the output file saved
    filename = 'C:Path//PAH[AvgAbsHours].csv'
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # 2. create a csvwriter object
        csvwriter.writerow(header[0])  # 4. write the header
        csvwriter.writerows(o_data_set)
    print("Created and saved database where null values are converted to column averages")


if __name__ == "__main__":
    #dapaPath needs the directory path to the Absenteeism_at_work_Project.csv file

    dataPath = 'C:Path//Absenteeism_at_work_Project.csv'
    print("Reading absentee data")

    createAvgSubData(dataPath)
    createZeroSubData(dataPath)
