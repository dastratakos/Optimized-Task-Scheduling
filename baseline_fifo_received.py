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
    assert(len(argv) >= 3)
    dataFile = argv[0]
    maxReqPerDay = int(argv[1])
    timeFrame = int(argv[2])
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
    racquets = []
    reqIDS = []
    for key in orderReceivedKeys:
        for reqID in orderingDateTime[key]:
            racquets.append
            racquetStr = str(requests[reqID][2])
            if requests[reqID][1] == 'True': racquetStr += 'SMT'
            else: racquetStr += 'Reg'
            daysUntilDue = (1 * (requests[reqID][2] == 'Exp')) + (3 * (requests[reqID][2] == 'Std'))
            racquets.append(((racquetStr, daysUntilDue), reqID, requests[reqID][3]))
            reqIDS.append(reqID)

    count = 0
    endDay = int(racquets[0][2]) - 1 + timeFrame
    currDay = int(racquets[0][2]) - 1
    dayList = []
    fulfilled = []
    for r in racquets:
        if count%maxReqPerDay == 0:
            if dayList: 
                dayList.sort(key = lambda x: x[0] + str(x[1]))
                print(dayList, '\n')
            currDay += 1
            if currDay > endDay: break
            print('\n', '-'*70, 'Day %d' %int(count/maxReqPerDay+1), '-'*70)
            dayList = []
        dayList.append((r[0][0], int(r[0][1]) - (int(currDay) - int(r[2]))))
        fulfilled.append(r[1])
        count += 1
        if int(count / maxReqPerDay) > timeFrame: break


    # i = 0
    # print('\n', '='*50, 'FIFO (date/time received) Baseline Ordering','='*50)
    # racquets = []
    # fulfilled = []
    # for key in orderReceivedKeys:
    #     for reqID in orderingDateTime[key]:
    #         if int(i / maxReqPerDay) > timeFrame: break
    #         if i % maxReqPerDay == 0:
    #             if racquets:
    #                 racquets.sort(key = lambda x: x[0][0] + str(x[0][1]))
    #                 print([r[0] for r in racquets])
    #             print('\n', '-'*70, 'Day %d' %int(i/maxReqPerDay+1), '-'*70)
    #             racquets = []

    #         racquetStr = str(requests[reqID][2])
    #         if requests[reqID][1] == 'True': racquetStr += 'SMT'
    #         else: racquetStr += 'Reg'
    #         daysUntilDue = (1 * (requests[reqID][2] == 'Exp')) + (3 * (requests[reqID][2] == 'Std'))
    #         racquets.append(((racquetStr, daysUntilDue), reqID))
    #         fulfilled.append(((racquetStr, daysUntilDue), reqID))
    #         i+=1
    #     if int(i / maxReqPerDay) > timeFrame: break
    # if racquets: 
    #     racquets.sort(key = lambda x: x[0][0] + str(x[0][1]))
    #     print([r[0] for r in racquets], '\n')

    # print('='*68, 'Complete', '='*68)
        


    # except:
    #     print('Invalid file name/format.')
