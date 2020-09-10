'''
File: Create Graph
----------
This file contains code for converting csv file data to a line graph
'''
import sys
import csv
import sklearn
import numpy as np
import math
import matplotlib.pyplot as plt

argv = sys.argv[1:]
print(argv[0])
assert len(argv) >= 1
assert argv[0][-4:] == '.csv'

f = open(argv[0], 'r')
fileReader = csv.reader(f)
x, y = [], []
for line in fileReader:
    x.append(int(line[0]))
    y.append(float(line[1]))

plt.scatter(x, y, marker='.')
# plt.title('Rewards Over Episodes\nDataset: ', argv[0],'\nConstant epsilon value: \u03B5=0.2')
# plt.title('Rewards Over Episodes\nDataset: ', argv[0],'\nEpsilon-decreasing Policy: \u03B5\N{SUBSCRIPT ZERO}=1.0')
# plt.title('Rewards Over Episodes\nDataset: ', argv[0],'\nEpsilon-decreasing Policy: \u03B5\N{SUBSCRIPT ZERO}=0.5')
plt.title('Rewards Over Episodes\nDataset: %s \nEpsilon-decreasing Policy: \u03B5\N{SUBSCRIPT ZERO}=0.2' %argv[0])

plt.xlabel('Episode number')
plt.ylabel('Reward')

plt.show()
