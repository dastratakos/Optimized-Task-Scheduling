'''
File: submission.py
----------
This file contains our implementations of value iteration and Q-learning.
'''

import util, math, random
from collections import defaultdict
from util import ValueIteration
from itertools import combinations

class RacquetsMDP(util.MDP):
    def __init__(self, numRacquets, numDays, file):
        """
        numRacquets: number of racquets that can be string in a day
        numDays: number of days to consider
        """
        self.numRacquets = numRacquets
        self.numDays = numDays
        self.data = readFile(file)
        # TODO: add variable for costs of stringing racquets

    # file is a string that is the name of the CSV data file
    # returns a data structure of racquets with their data, grouped by day
    def readFile(file):
        f = open(file, 'r') # to read the file
        fileReader = csv.reader(f)
        data = []
        day = []
        for lineNum, row in enumerate(fileReader):
            if lineNum == 0:
                continue
            elif lineNum == 1:
                day.append((row[0], row[1], row[2], row[3], row[4]))
            else:
                if row[3] == day[len(day) - 1][3]:
                    day.append((row[0], row[1], row[2], row[3], row[4]))
                else:
                    data.append(day)
                    day = [(row[0], row[1], row[2], row[3], row[4])]
        data.append(day)
        return tuple(data)

    # Empty list of racquets at the start of Day 0
    def startState(self):
        return ([], 0)

    # Return a list of strings representing actions possible from |state|.
    # One action is a tuple of indices that represent picking any 15 (or less) racquets.
    def actions(self, state):
        if len(state[0]) < self.numRacquets: # all racquets can be strung for that day
            return range(len(state[0])) # return list of indices of all racquets
        # otherwise, there are more racquets to string that can be strung for that day
        return combinations(range(len(state[0])), self.numRacquets)

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        if state[1] == self.numDays: return [] # end state
        
        # remove racquets based on the action and compute reward of stringing those racquets
        # (add in probability of customer unsatisfied -> transition probabilities)
        
        # add new racquets for next day
        # (generate new data -> transition probabilities?)
        
        return [(1, 0.15, 200), (-1, 0.85, 10)]

    # Set the discount factor (float or integer).
    def discount(self):
        return 1
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
