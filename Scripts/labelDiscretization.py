import csv
import math

#This script converts the continuous Absentee Hour values into discrete labels
# 0: A, 1-15: B, 16-120: C

def readCSV(filePath, head, DataSet):
    file = open(filePath)
    csvreader = csv.reader(file)
    #next(csvreader)
    head.append(next(csvreader))
    #index = 10
    for line in csvreader:
        DataSet.append(line)
        #if index > 0:
        #    oDataSet.append(line)
        #    index -= 1
    file.close()

def convertToDiscrete(table):
    for row in table:
        #We are only accessing column 20 because that is where the absentee hour values are
        if row[20].strip().isnumeric():
            if int(row[20]) == 0:
                row[20] = "A"
            elif 1 <= int(row[20]) <= 15:
                row[20] = "B"
            else:
                row[20] = "C"
    #prints the first row of data. Final column data value should be A, B, or C
    #print(table[1])
    #print("Label discretization complete")

def discretizeZeroDB():
    # enter the path where the input file is located and where you want the new output file to be saved
    input = 'C:Path//PAH[ZeroAbsHours].csv'
    output = 'C:Path/PAH[ZeroAB_Labeled].csv'

    header = []
    data_set = []

    readCSV(input, header, data_set)
    convertToDiscrete(data_set)

    with open(output, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # 2. create a csvwriter object
        csvwriter.writerow(header[0])  # 4. write the header
        csvwriter.writerows(data_set)


def discretizeAvgDB():
    # enter the path where the input file is located and where you want the new output file to be saved
    input = 'C:Path//PAH[AvgAbsHours].csv'
    output = 'C:Path/PAH[AvgAB_Labeled].csv'

    header = []
    data_set = []

    readCSV(input, header, data_set)
    convertToDiscrete(data_set)

    with open(output, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # 2. create a csvwriter object
        csvwriter.writerow(header[0])  # 4. write the header
        csvwriter.writerows(data_set)


if __name__ == "__main__":
    discretizeAvgDB()
    discretizeZeroDB()
    print("Completed Table Discretization")