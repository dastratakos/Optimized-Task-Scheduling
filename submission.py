'''
file: submission.py
authors: Kento Perera, Timothy Sah, and Dean Stratakos
date: December 1, 2019
----------
This file contains our implementations of Value Iteration
and Q-learning for our CS 221 Project.
We borrowed code structure from blackjack/submission.py
'''

import util, math, random, csv, timeit
from collections import defaultdict
from util import ValueIteration
from itertools import combinations

'''
class: RacquetsMDP
----------
Defines the MDP for the racquet stringing problem.
'''
class RacquetsMDP(util.MDP):
    '''
    function: __init__
    ----------
    Constructor for the RacquetsMDP class.
        numRacquets: number of racquets that can be string in a day
        file: a string that is the name of the CSV data file
        numDays: number of days to consider
        
        data: tuple of racquet job request data separated by day
    '''
    def __init__(self, numRacquets, file, numDays):
        self.numRacquets = numRacquets
        self.data = self.readFile(file)
        self.numDays = min(numDays, len(self.data))
    
    '''
    function: readFile
    ----------
    Parses an input CSV file and stores the contents in self.data
        file: a string that is the name of the CSV data file
    returns data, which is a tuple of lists; day[i] is a list representing
    the racquet requests for day i + 1 as a list of tuples; day[i][j] is a
    tuple representing the (j + 1)th racquet request on day i + 1
    '''
    def readFile(self, file):
        f = open(file, 'r') # to read the file
        fileReader = csv.reader(f)
        data = []
        day = []
        currDate = 0
        for lineNum, row in enumerate(fileReader):
            daysUntilDue = (1 * (row[2] == 'Exp')) + (3 * (row[2] == 'Std'))
            reqType = row[2]                        # to build request string
            if row[1] == 'TRUE': reqType += 'SMT'   # to build request string
            else: reqType += 'Reg'                  # to build request string
            
            if lineNum == 0:
                continue
            elif lineNum == 1:
                day.append((reqType, daysUntilDue))
                currDate = row[3]
            else:
                if row[3] == currDate:
                    day.append((reqType, daysUntilDue))
                else:
                    data.append(day)
                    day = []
                    day.append((reqType, daysUntilDue))
                    currDate = row[3]
        data.append(day)
        return data
        
    '''
    function: startState
    ----------
    The start state contains a tuple of the racquets at the start of Day 1
    and an integer indicating that the state is at the start of Day 1.
    '''
    def startState(self):
        return ((tuple(self.data[0]), 1))

    '''
    function: actions
    ----------
    Return a list of lists representing actions possible from |state|.
    One action is a list of racquets that represent picking any
    self.numRacquets (or fewer) racquets to string for the current day.
    '''
    def actions(self, state):
        if state == (): return [0]
        if len(state[0]) < self.numRacquets: # all racquets can be strung for that day
            return [state[0]] # return list of all racquets
        # otherwise, there are more racquets to string that can be strung for that day
        return set(combinations(state[0], self.numRacquets))

    '''
    function: succAndProbReward
    ----------
    Given a |state| and |action|, returns a list of (newState, prob, reward) tuples
    corresponding to the states reachable from |state| when taking |action|.
    If |state| is an end state, returns an empty list [].
    '''
    def succAndProbReward(self, state, action, bounded=True, bound=6):
        # end state when we have processed the last day
        if state[1] == self.numDays + 1: return []
        
        racquets = list(state[0])
        
        # remove racquets based on the action and compute reward of stringing those racquets
        for racquet in action:
            racquets.remove(racquet)
        
        # decrement days until due for remaining racquets
        for i in range(len(racquets)):
            racquet = racquets[i]
            racquets[i] = (racquet[0], racquet[1] -  1)
        
        # add new racquets for next day
        if state[1] <= len(self.data) - 1:
            for racquet in self.data[state[1]]:
                # sets upper bound if too many requests built up
                if bounded and len(racquets) >= self.numRacquets + bound:
                    break
                racquets.append(racquet)
        racquets.sort(key = lambda x: x[0] + str(x[1]))
            
        # compute reward in $
        #   $20 penalty if racquet will be overdue
        #   $10 penalty if racquet will be overdue in following day
        # if requests are same type, then break the tie by assigning slightly higher reward for stringing the older one
        reward = 0
        for racquet in action:
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
                reward += (18 + (3 - racquet[1])*.01)
                
            # penalize unstrung racquets if they are overdue today or tomorrow
            for racquet in racquets:
                if (racquet[1] < 0): reward += (20 * racquet[1])
                if (racquet[1] - 1 < 0): reward += (10 * racquet[1] - 1)
            
        return [((tuple(racquets), state[1] + 1), 1, reward)]

    '''
    function: discount
    ----------
    Sets the discount factor.
    '''
    def discount(self):
        return 1

'''
function: identityFeatureExtractor
----------
Returns a single-element list containing a binary (indicator) feature
for the existence of the (racquets, action) pair.  Provides no generalization.
'''
def identityFeatureExtractor(state, action):
    featureKey = (tuple(state[0]), action)
    featureValue = 1
    return featureKey, featureValue

'''
class: QLearningAlgorithm
----------
Defines the Q-learning algorithm. More information in util.RLAlgorithm.
'''
class QLearningAlgorithm(util.RLAlgorithm):
    '''
    function: __init__
    ----------
    Construtor for QLearningAlgorithm.
        actions: a function that takes a state and returns a list of actions.
        discount: a number between 0 and 1, which determines the discount factor
        featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
        explorationProb: the epsilon value indicating how frequently the policy
        returns a random action
    '''
    def __init__(self, actions, discount, featureExtractor=identityFeatureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.qStarActions = defaultdict(list)

    '''
    function: getQ
    ----------
    Returns the Q function associated with the weights and features
    '''
    def getQ(self, state, action):
        score = 0.0
        f, v = self.featureExtractor(state, action)
        score += self.weights[tuple(f)] * v
        return score
    
    '''
    function: getAction
    ----------
    This algorithm will produce an action given a state.
    Here we use the epsilon-greedy algorithm: with probability
    |explorationProb|, take a random action.
    '''
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(list(self.actions(state)))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    '''
    function: getStepSize
    ----------
    Returns the step size to update the weights.
    '''
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    '''
    function: incorporateFeedback
    ----------
    This function is called by util.py with (s, a, r, s'), which is used to update |weights|.
    '''
    def incorporateFeedback(self, state, action, reward, newState):
        target = reward
        if newState is not None:
            qOpt = [self.getQ(newState, action) for action in self.actions(newState)]
            target += self.discount * max(qOpt)
        prediction = self.getQ(state, action)
        name, value = self.featureExtractor(state, action)
        self.weights[name] -= self.getStepSize() * (prediction - target) * value

'''
function: testValueIteration
----------
Test function for Value Iteration.
'''
def testValueIteration(mdp):
    valueIter = ValueIteration() # implemented in util.py
    valueIter.solve(mdp, .001)
    states = sorted(valueIter.pi, key=lambda x: len(x)) # sorted by state space
    
    print('valueIter.pi:')
    for elem in sorted(valueIter.pi):
        print(elem, '\t:\t', valueIter.pi[elem])
        
    return valueIter

'''
function: testQLearning
----------
Test function for Q-Learning.
'''
def testQLearning(mdp):
    qLearn = QLearningAlgorithm(mdp.actions, mdp.discount())
    rewards = util.simulate(mdp, qLearn, 500)
    # for i in range(0,300,25):
    #     print('Average reward, episodes %d - %d: %d' %(i, i+25, sum(rewards[i:i+25]) / 25))    
    qLearn.explorationProb = 0
    
    print('qLearn.qStarActions:')
    for elem in sorted(qLearn.qStarActions):
        print(elem, '\t:\t', qLearn.qStarActions[elem])
        
    return qLearn

'''
function: compareResults
----------
Compares the results of Value Iteration and Q-Learning.
'''
def compareResults(valueIter, qLearn):
    diff = 0.0
    for state in valueIter.pi:
        if qLearn.qStarActions[state] != [] and valueIter.pi[state] != qLearn.qStarActions[state][0]:
            diff += 1
        elif qLearn.qStarActions[state] != [] and valueIter.pi[state] == qLearn.qStarActions[state][0]:
            print('Same policy mapping \n\t STATE---', state, '\n\t\t--- to action ---', valueIter.pi[state])
    print('Number of different policy instructions: ', diff)
    print('Length of pi_valueIter: ', len(valueIter.pi))
    print('Length of pi_QStar: ', len(qLearn.qStarActions))
    print('Difference over length of pi_valueIter:', diff / len(valueIter.pi))
    print('Difference over length of pi_QStar:', diff / len(qLearn.qStarActions))

'''
function: main
----------
Initializes an MDP and runs appropriate algorithms.
'''
def main():
    start = timeit.default_timer()
    valueIteration = False
    qLearning = True

#    mdp = RacquetsMDP(4, 'test_data_save.csv', 6)
#    mdp = RacquetsMDP(15, 'training_data.csv', 10)
#    mdp = RacquetsMDP(13, 'training_data_small.csv', 10)
    # mdp = RacquetsMDP(13, 'training_data_big.csv', 6)
    mdp = RacquetsMDP(13, 'training_data_1211_1.csv', 12)
    if valueIteration:
        valueIter = testValueIteration(mdp)
    if qLearning:
        qLearn = testQLearning(mdp)
    if valueIteration and qLearning:
        compareResults(valueIter, qLearn)
    
    stop = timeit.default_timer()
    print('\nTime:', stop - start, 'sec')
    
if __name__ == '__main__':
    main()
