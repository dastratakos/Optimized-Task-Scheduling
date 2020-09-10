'''
Generates data and writes it into a csv file called [FILE_NAME]
'''
import sys
import csv
import sklearn
import numpy as np
import util, math, random, collections

# FILENAME to write to
FILE_NAME = 'training_data_big.csv' 

# Data labels
# 	Request ID: 		Integer 	|| range(0, len(numRequests))
#	SMT: 				Boolean   	|| [True, False]
#	Service Requested: 	List(str)   || ['Standard' (3 days), 'Express' (24 hours), 'Speedy' (while you wait)]
#	Date: 				int 		|| year month day
#	Time: 				int  		|| hour minute
DATA_LABELS = ['Request ID', 'SMT', 'Service Requested', 'Date', 'Time']

REQ_ID_INDEX = 0    # Integer     	|| range(0, len(numRequests))
SMT_BOOLEAN = 1     # Boolean     	|| [True, False]
SERVICE_INDEX = 2   # List(str)   	|| ['Standard' (3 days), 'Express' (24 hours), 'Speedy' (while you wait)]
DATE_INDEX = 3      # int  			|| year month day
TIME_INDEX = 4      # int  			|| hour minute

# Constant values/probabilities
SMT_FREQ = 0.14487808
SMT_FREQ_IN_SEASON = 0.3
SUNDAY_HOURS = 5
REGULAR_HOURS = 9
REQUESTS_PER_WEEK = 93.1818
REQUESTS_PER_HOUR = REQUESTS_PER_WEEK / ((REGULAR_HOURS*6) + (SUNDAY_HOURS*1))
REGULAR_REQ_WEIGHTS = {'Std': 0.9, 'Exp': 0.09, 'Spd': 0.01}
SMT_REQ_WEIGHTS = {'Spd': 0.1, 'Exp': 0.35, 'Std': 0.55}

DEFAULT_GENERATE_NUMHOURS = 9
DEFAULT_NUMDAYS = 5
DEFAULT_STARTDATE = (1, 1, 2019)
DEFAULT_IN_SEASON = True
CALENDAR_DICT = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}



def generateData(numHours, numDays, startDate, inSeason):
	print('Number of hours of data to generate: ', numHours)
	print('In season: ', inSeason)

	f = open(FILE_NAME, 'w')
	w = csv.writer(f)
	w.writerow(DATA_LABELS)

	smtFreq = SMT_FREQ
	if inSeason: smtFreq = SMT_FREQ_IN_SEASON
	print(smtFreq)
	date = [startDate[0], startDate[1], startDate[2]]
	reqID_Count = 0
	for day in range(numDays):
		reqList = np.random.poisson(REQUESTS_PER_HOUR, numHours)		# reqList is a list where each index (hourNum) contains number of requests during that hour
		print('------------------------------')
		print('Day number:', day + 1)
		print('Request Breakdown: ', reqList)
		print('Total Number of Requests in %d hours: %d' % (numHours, sum(reqList)))

		date[1] += 1
		for hour, reqAtHour in enumerate(reqList):
			for i in range(reqAtHour):
				# Request ID Index
				reqID_Count += 1

				# SMT
				StanfordMensTennis = (random.random() < smtFreq)

				# Service Requested
				serviceReq = None
				if StanfordMensTennis: serviceReq = util.weightedRandomChoice(SMT_REQ_WEIGHTS)
				else: serviceReq = util.weightedRandomChoice(REGULAR_REQ_WEIGHTS)

				# Date
				if date[1] > CALENDAR_DICT[date[0]]:
					date[1] = date[1] - CALENDAR_DICT[date[0]]
					date[0] += 1
					if date[0] == 13: 
						date[0] = 1
						date[2] += 1
				month = str(date[0])
				day = str(date[1])
				year = str(date[2])
				if len(month) == 1: month = '0' + month
				if len(day) == 1: day = '0' + day
				while len(year) < 4: year = '0' + year
				dateInt = int(year + month + day)


				# Time
				minutes = random.randint(0, 60)
				minutesStr = str(minutes)
				hourStr = str(hour)
				if minutes <= 10: minutesStr = '0' + minutesStr
				if len(hourStr) == 1: hourStr = '0' + hourStr
				time = hourStr + minutesStr
				timeTup = (hour, minutes)

				print('RequestID: %d || SMT: %s || Service Requested: %s || Date: %s || Time: %s' %(reqID_Count, str(StanfordMensTennis), str(serviceReq), str(date[0]) + '/' + str(date[1]) + '/' + str(date[2]), str(timeTup)))

				newData = [reqID_Count, StanfordMensTennis, serviceReq, dateInt, time]
				w.writerow(newData)

	f.close()

def dateCorrectlyFormatted(dateEntry):
	date = dateEntry.split('/')
	print(date)
	if len(date) == 3:
		year = date[2]
		month = date[0]
		day = date[1]
		if (not year.isdigit()) or (not month.isdigit()) or (not day.isdigit()): return False, (None, None, None)
		year = int(year)
		month = int(month)
		day = int(day)
		if CALENDAR_DICT[month] == None or day < 1 or day > CALENDAR_DICT[month]: return False, (None, None, None)
		return [True, (month, day, year)]
	return False, (None, None, None)



if __name__ == '__main__':
	print('====GENERATING DATA====')
	argv = sys.argv[1:]
	if len(sys.argv) == 0: 
		generateData(DEFAULT_GENERATE_NUMHOURS, DEFAULT_NUMDAYS, DEFAULT_STARTDATE, DEFAULT_IN_SEASON)
	elif len(argv) == 1 and argv[0].isdigit(): 
		generateData(int(argv[0], DEFAULT_NUMDAYS, DEFAULT_STARTDATE, DEFAULT_IN_SEASON))
	elif len(argv) == 2 and (argv[0].isdigit()) and (argv[1].isdigit()): 
		generateData(int(argv[0]), int(argv[1]), DEFAULT_STARTDATE, DEFAULT_IN_SEASON)
	elif len(argv) == 3 and (argv[0].isdigit()) and (argv[1].isdigit()) and (dateCorrectlyFormatted(argv[2])[0]): 
		generateData(int(argv[0]), int(argv[1]), dateCorrectlyFormatted(argv[2])[1], DEFAULT_IN_SEASON)
	elif len(argv) >= 4 and (argv[0].isdigit()) and (argv[1].isdigit()) and (dateCorrectlyFormatted(argv[2])[0]) and (argv[3] == 'True' or 'False'): 
		generateData(int(argv[0]), int(argv[1]), dateCorrectlyFormatted(argv[2])[1], argv[3] == 'True')
	else:
		print('Invalid arguments. Arguments must follow the format: python3 data_generator integer True/False')
	print('====DATA GENERATED====')

    
    
    
