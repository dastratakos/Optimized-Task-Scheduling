'''
File: Oracle (FIFO, date/time received)
----------
This file contains code for our FIFO (received order) oracle, which reads in a csv file containing data and selects
rackets to string in a FIFO order of when they were received.
'''
import sys
import csv
import sklearn
import numpy as np
import util, math, random, collections

REQ_ID_INDEX = 0    # Integer     || range(0, len(numRequests))
SMT_BOOLEAN = 1     # Boolean     || [True, False]
SERVICE_INDEX = 2   # List(str)   || ['Standard' (3 days), 'Express' (24 hours), 'Speedy' (while you wait)]
DATE_INDEX = 3      # int  || year month day
TIME_INDEX = 4      # int  || hour minute


if __name__ == '__main__':
    argv = sys.argv[1:]
    dataFile = argv[0]
    requests = {}
    orderingDateTime = collections.defaultdict(list)
    # try:
    f = open(dataFile, 'r')
    reader = csv.reader(f)
    lineNum = 0
    for row in reader:
        if lineNum == 0:
            # print(f'Column names are {",".join(row)}')
            lineNum = 0
        else:
            requests[int(row[0])] = (row[0], row[1], row[2], row[3], row[4])
            # print('RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])))
            date = row[3]
            time = row[4]
            reqId = row[0]
            orderingDateTime[date + time].append(int(row[0]))
        lineNum+=1

    orderReceivedKeys = list(orderingDateTime.keys())
    orderReceivedKeys.sort()
    i = 1
    print('==========FIFO (date/time received) Baseline Ordering==========')
    for key in orderReceivedKeys:
        for reqID in orderingDateTime[key]:
            print('Priority number %d ----- RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(i, int(requests[reqID][0]), str(requests[reqID][1]), str(requests[reqID][2]), str(requests[reqID][3]), str(requests[reqID][4])))
            i += 1
    print('==========================Complete==========================')
        


    # except:
    #     print('Invalid file name/format.')
