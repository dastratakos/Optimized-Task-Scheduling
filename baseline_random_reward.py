'''
file: fifo_reward.py
authors: Kento Perera, Timothy Sah, and Dean Stratakos
date: December 13, 2019
----------
This file contains our implementations of calculating total reward via our baseline FIFO algorithm
'''

import util, math, random, csv, timeit
from collections import defaultdict
from util import ValueIteration
from itertools import combinations


'''
function: main
----------
Calculates the reward using FIFO algorithm.
'''
def main():
    start = timeit.default_timer()
    
    f = open("training_data_TEST2.csv", 'r') # to read the file

    fileReader = csv.reader(f)
    data = []
    day = []
    currDate = 0
    bound = 7
    capacity = 13
    for lineNum, row in enumerate(fileReader):
        if lineNum == 0:
            continue
    
        time = row[4]
        if int(row[4]) < 10: timeStamp = row[3] + '00' + row[4]
        elif int(row[4]) < 100: timeStamp = row[3] + '0' + row[4]
        else: timeStamp = row[3] + row[4]
        
        daysUntilDue = (1 * (row[2] == 'Exp')) + (3 * (row[2] == 'Std'))
        reqType = row[2]                        # to build request string
        if row[1] == 'True': reqType += 'SMT'   # to build request string
        else: reqType += 'Reg'                  # to build request string
        
        if lineNum == 1:
            day.append([reqType, daysUntilDue, int(timeStamp)])
            currDate = row[3]
            howManyRacquetsSeenInDay = 1
        else: # if not on first racquet
            if row[3] == currDate: # if current racquet is still on same day
                if howManyRacquetsSeenInDay < (capacity + bound):
                    howManyRacquetsSeenInDay += 1
                    day.append([reqType, daysUntilDue, int(timeStamp)])
            else: # if current racquet is on a new day
                data.append(sorted(day, key=lambda x: x[2]))
                day = []
                day.append([reqType, daysUntilDue, int(timeStamp)])
                currDate = row[3]
                howManyRacquetsSeenInDay = 1
    data.append(sorted(day, key=lambda x: x[2]))
    reward = 0
    strung = []
    unstrung = []
    for dayNumber in range(len(data)): # iterates for number of days in time frame
        unstrung.extend(data[dayNumber])
        index = 0
        # calculates revenue for completed racquets
        while index < capacity and len(unstrung) > 0: # while you can still string the racquet and there are racquets left to accept
            racquet = unstrung.pop(random.randint(0, len(unstrung) - 1))
            index += 1
            strung.append(racquet)
            
            # calculate reward
            if racquet[0] == 'SpdReg':
                reward += 40
            elif racquet[0] == 'ExpReg':
                reward += (30 + (1 - racquet[1]) * .01)
            elif racquet[0] == 'StdReg':
                reward += (20 + (3 - racquet[1]) * .01)
            elif racquet[0] == 'SpdSMT':
                reward += 18
            elif racquet[0] == 'ExpSMT':
                reward += (18 + (1 - racquet[1]) * .01)
            elif racquet[0] == 'StdSMT':
                reward += (18 + (3 - racquet[1]) * .01)

        for racquet in unstrung:
            if (racquet[1] < 0): reward += (20 * racquet[1])
            if (racquet[1] - 1 < 0): reward += (10 * (racquet[1] - 1))
            if dayNumber != len(data) - 1:
                racquet[1] -= 1
                data[dayNumber + 1].append(racquet)
            
        print('=' * 30, 'DAY', dayNumber, '=' * 30)
        for racquet in strung:
            print(racquet)
            
    print('Reward: ', reward)
 
    stop = timeit.default_timer()
    print('\nTime:', stop - start, 'sec')
    
if __name__ == '__main__':
    main()

