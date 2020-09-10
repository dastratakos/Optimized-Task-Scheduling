'''
File: Oracle (FIFO, date/time due)
----------
This file contains code for our FIFO (due date) oracle, which reads in a csv file containing data and selects
rackets to string in a FIFO order of when they were received.
'''
import sys
import csv
import sklearn
import numpy as np
import util, math, random, collections

REQ_ID_INDEX = 0    # Integer       || range(0, len(numRequests))
SMT_BOOLEAN = 1     # Boolean       || [True, False]
SERVICE_INDEX = 2   # List(str)     || ['Standard' (3 days), 'Express' (24 hours), 'Speedy' (while you wait)]
DATE_INDEX = 3      # int           || year month day
TIME_INDEX = 4      # int           || hour minute

CALENDAR_DICT = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


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
            dueDateIncrement = 0
            if row[SERVICE_INDEX] == 'Standard': dueDateIncrement = 3
            if row[SERVICE_INDEX] == 'Express': dueDateIncrement = 1
            if row[SERVICE_INDEX] == 'ASAP': dueDateIncrement = 0
            dateStr = str(row[DATE_INDEX])
            year = int(dateStr[0:4])
            month = int(dateStr[4:6])
            day = int(dateStr[6:])
            day += dueDateIncrement
            if day > CALENDAR_DICT[month]:
                day = day - CALENDAR_DICT[month]
                month += 1
                if month == 13: 
                    month = 1
                    year += 1
            month = str(month)
            day = str(day)
            year = str(year)
            if len(month) == 1: month = '0' + month
            if len(day) == 1: day = '0' + day
            while len(year) < 4: year = '0' + year
            dateInt = int(year + month + day)


            requests[int(row[0])] = (row[0], row[1], row[2], row[3], row[4])
            # print('RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])))
            date = str(dateInt)
            time = row[4]
            reqId = row[0]
            orderingDateTime[date + time].append(int(row[0]))
        lineNum+=1

    orderReceivedKeys = list(orderingDateTime.keys())
    orderReceivedKeys.sort()
    i = 1
    print('==========FIFO (based on due date) Oracle Ordering==========')
    for key in orderReceivedKeys:
        for reqID in orderingDateTime[key]:
            print('Priority number %d -----Due on %s at %s----- RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(i, key[0:8], key[8:], int(requests[reqID][0]), str(requests[reqID][1]), str(requests[reqID][2]), str(requests[reqID][3]), str(requests[reqID][4])))
            i += 1
    print('==========================Complete==========================')

    # except:
    #     print('Invalid file name/format.')
