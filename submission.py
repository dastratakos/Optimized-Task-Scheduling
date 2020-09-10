'''
File: submission.py
----------
This file contains our implementations of value iteration and Q-learning.
'''

import util, math, random, csv
from collections import defaultdict
from util import ValueIteration
from itertools import combinations

class RacquetsMDP(util.MDP):
    def __init__(self, numRacquets, file, numDays, returnProb):
        """
        numRacquets: number of racquets that can be string in a day
        numDays: number of days to consider
        data: tuple of racquet job request data
        returnProb: probability that a customer is unsatisfied with job
        """
        self.numRacquets = numRacquets
        self.data = self.readFile(file)
        self.numDays = min(numDays, len(self.data))
        self.returnProb = returnProb
        # TODO: add variable for costs of stringing racquets

    # file is a string that is the name of the CSV data file
    # returns a data structure of racquets with their data, grouped by day
    # data structure is a tuple of lists; each list represents one day of
    #   racquet intakes as a list of tuples; each tuple is a racquet
    def readFile(self, file):
        f = open(file, 'r') # to read the file
        fileReader = csv.reader(f)
        data = []
        day = set()
        currDate = 0
        for lineNum, row in enumerate(fileReader):
            if lineNum == 0:
                continue
            elif lineNum == 1:
                day.add((row[0], row[1], row[2], row[3], row[4]))
                currDate = row[3]
            else:
                if row[3] == currDate:
                    day.add((row[0], row[1], row[2], row[3], row[4]))
                else:
                    data.append(day)
                    day = set((row[0], row[1], row[2], row[3], row[4]))
                    currDate = row[3]
        data.append(day)
#        return tuple(data)
        return data

    # Start state is an empty list of racquets at the start of Day 0
    def startState(self):
        return ((), 0)

    '''
    NOTE: might need to check for empty list of racquets
    '''
    # Return a list of lists representing actions possible from |state|.
    # One action is a list of racquet IDs that represent picking any 15 (or less) racquets.
    def actions(self, state):
        if len(state[0]) < self.numRacquets: # all racquets can be strung for that day
            return [state[0]] # return list of IDs of all racquets
        # otherwise, there are more racquets to string that can be strung for that dayp
        return combinations(state[0], self.numRacquets)

    # TODO: add a count of number of racquets rejected, then compute probability of that happening
    # Given a |state| and |action|, returns a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # If |state| is an end state, returns an empty list [].
    def succAndProbReward(self, state, action):
        if state[1] == self.numDays: return [] # end state
        
        racquetIDs = set(state[0])
        
        # remove racquets based on the action and compute reward of stringing those racquets
        racquetIDs -= set(action)
        
        # TODO: (add in probability of customer unsatisfied -> transition probabilities)
        
        # add new racquets for next day
        # (generate new data -> transition probabilities?)
        racquetIDs |= self.data[state[1]]
        return [((tuple(racquetIDs), state[1] + 1), 1, 1000)]

    # Set the discount factor (float or integer).
    def discount(self):
        return 1
        
def testMDP():
    mdp = RacquetsMDP(4, 'test_data_save.csv', 6, 0)
    mdp.computeStates()
    print('*' * 30, 'middle of code', '*' * 30)
    algorithm = ValueIteration() # implemented for us in util.py
    print('*' * 30, 'you suck', '*' * 30)
    algorithm.solve(mdp, .001)

print('start of code')
testMDP()
print('end of code')

'''
############################################################
# Reference Blackjack Problem 4b: convergence of Q-learning
# Small test case
# smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
smallMDP = RacquetsMDP()

# Large test case
#largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP = RacquetsMDP()

def simulate_QL_over_MDP(mdp, featureExtractor):
    # Q-learning
    mdp.computeStates()
    qLearn = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    util.simulate(mdp, qLearn, 30000)
    
    # value iteration
    mdp.computeStates()
    valueIter = ValueIteration()
    valueIter.solve(mdp)
    
    qLearn.explorationProb = 0

    # compare
    diff = 0
    for state in valueIter.pi:
        if valueIter.pi[state] != qLearn.getAction(state):
            diff += 1
    print('Difference:', diff / len(valueIter.pi))
    # END_YOUR_CODE
'''
