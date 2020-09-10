'''
File: Oracle (random choice)
----------
This file contains code for our oracle, which reads in a csv file containing data and randomly selects
rackets to string in a random order, as if all the rackets were placed in a large bucket arbitrarily.
'''
import sys
import csv
import sklearn
import numpy as np
import util, math, random, collections


# Need to include due date/time in the print statement (based on )

if __name__ == '__main__':
    argv = sys.argv[1:]
    dataFile = argv[0]
    requests = []
    try:
        f = open(dataFile, 'r')
        reader = csv.reader(f)
        lineNum = 0
        for row in reader:
            if lineNum == 0:
                # print(f'Column names are {",".join(row)}')
                lineNum = 0
            else:
                requests.append((int(row[0]), row[1], row[2], row[3], row[4]))
                # print('RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])))
            lineNum+=1
        # print(requests)

        randIndex = list(range(len(requests)))
        random.shuffle(randIndex)
        print('==============Random Choice Oracle Ordering==============')
        for i in range(len(requests)):
            print('Priority number %d ----- RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(i+1, int(requests[randIndex[i]][0]), str(requests[randIndex[i]][1]), str(requests[randIndex[i]][2]), str(requests[randIndex[i]][3]), str(requests[randIndex[i]][4])))
        print('==========Random Choice Oracle Ordering Complete==========')

    except:
        print('Invalid file name/format.')
