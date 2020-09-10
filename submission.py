'''
File: submission.py
----------
This file contains our implementations of value iteration and Q-learning.
'''

import util, math, random, csv, timeit
from collections import defaultdict
from util import ValueIteration
from itertools import combinations

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class RacquetsMDP(util.MDP):
    def __init__(self, numRacquets, file, numDays, rejectProb):
        """
        numRacquets: number of racquets that can be string in a day
        numDays: number of days to consider
        data: tuple of racquet job request data
        returnProb: probability that a customer is unsatisfied with job
        """
        self.numRacquets = numRacquets
        self.data = self.readFile(file)
        self.numDays = min(numDays, len(self.data))
        self.rejectProb = rejectProb
        # TODO: add variable for costs of stringing racquets

    # file is a string that is the name of the CSV data file
    # returns a data structure of racquets with their data, grouped by day
    # data structure is a tuple of lists; each list represents one day of
    #   racquet intakes as a list of tuples; each tuple is a racquet
    def readFile(self, file):
        f = open(file, 'r') # to read the file
        fileReader = csv.reader(f)
        data = []
        day = []
        currDate = 0
        for lineNum, row in enumerate(fileReader):
            daysUntilDue = (1*(row[2] == 'Exp')) + (3*(row[2] == 'Std'))
            reqType = row[2]
            if row[1] == 'TRUE': reqType += 'SMT'
            else: reqType += 'Reg'
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
#        return tuple(data)
        print('=' * 36, ' data  ', '=' * 37)
        for n, d in enumerate(data): print(n, ' : ', d)
        return data

    # Start state is an empty list of racquets at the start of Day 0
    def startState(self):
        return ((), 0)

    '''
    NOTE: might need to check for empty list of racquets
    '''
    # Return a list of lists representing actions possible from |state|.
    # One action is a list of racquets that represent picking any self.numRacquets (or fewer) racquets.
    def actions(self, state):
        if len(state[0]) < self.numRacquets: # all racquets can be strung for that day
            return [state[0]] # return list of all racquets
        # otherwise, there are more racquets to string that can be strung for that day
        return combinations(state[0], self.numRacquets)

    # TODO: add a count of number of racquets rejected, then compute probability of that happening
    # Given a |state| and |action|, returns a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # If |state| is an end state, returns an empty list [].
    def succAndProbReward(self, state, action):
        print('succAndProbReward with', len(state[0]), 'racquets')
        # end state when we reach the end of the window of days
        if state[1] == self.numDays + 1: return []
    
        results = []
        sum = 0
        # for every possibility of number of rejected racquets
        for numRejected in range(len(list(state[0])) + 1):
            # total probability of having numRejected rejections
            probability = choose(len(list(state[0])), numRejected) *  ((self.rejectProb) ** numRejected) * ((1 - self.rejectProb) ** (len(list(state[0])) - numRejected))
            sum += probability
            print('\t', 'probability of', numRejected, 'rejected is', probability)
            
            if probability < 0.01: continue
            
            # for every combination of strung racquets
            for strung in list(combinations(list(state[0]), len(list(state[0])) - numRejected)):
                length = len(list(combinations(list(state[0]), len(list(state[0])) - numRejected)))
                racquets = list(state[0])
                # remove racquets based on the action and compute reward of stringing those racquets
                for racquet in strung:
                    racquets.remove(racquet)
                
                # decrement days until due for remaining racquets
                for i in range(len(racquets)):
                    racquet = racquets[i]
                    racquets[i] = (racquet[0], racquet[1] -  1)
                
                # add new racquets for next day
                # (generate new data -> transition probabilities?)
                if state[1] <= len(self.data) - 1:
                    racquets += self.data[state[1]]
#                if state[1] <= len(self.data) - 1:
#                    for racquet in self.data[state[1]]:
#                        racquets.append(racquet)
                    
                # compute reward in $, $2 penalty for each day late
                reward = 0
                for racquet in strung:
                    if racquet[0] == 'SpdReg': reward += 40
                    elif racquet[0] == 'ExpReg': reward += 30
                    elif racquet[0] == 'StdReg': reward += 20
                    elif racquet[0] == 'SpdSMT': reward += 18
                    elif racquet[0] == 'ExpSMT': reward += 18
                    elif racquet[0] == 'StdSMT': reward += 18
                    if racquet[1] < 0: reward += -2 * racquet[1]
                results.append(((tuple(racquets), state[1] + 1), probability / length, reward))
                
        print('  total probability:', sum)
#        return [((tuple(racquets), state[1] + 1), 1, reward)]
        return results

    # Set the discount factor (float or integer).
    def discount(self):
        return 1

'''
#############################################################################################################################

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        target = reward
        if newState is not None:
            qOpt = [self.getQ(newState, action) for action in self.actions(newState)]
            target += self.discount * max(qOpt)
        prediction = self.getQ(state, action)
        for name, value in self.featureExtractor(state, action):
            self.weights[name] -= self.getStepSize() * (prediction - target) * value
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Reference Blackjack Problem 4b: convergence of Q-learning
# Small test case
# smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
#smallMDP = RacquetsMDP()

# Large test case
#largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
#largeMDP = RacquetsMDP()

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




def testMDP():
    mdp = RacquetsMDP(4, 'test_data_save.csv', 6, 0.10)
    mdp.computeStates()
    algorithm = ValueIteration() # implemented for us in util.py
    algorithm.solve(mdp, .001)
    print('*' * 60)
    for state in algorithm.pi:
        print('state:', state)
        print('\toptimal action:', algorithm.pi[state])
    '''
    mdp.computeStates()
    qLearn = QLearningAlgorithm(mdp.actions, mdp.discount(), identityFeatureExtractor)
    util.simulate(mdp, qLearn, 30000)
    '''
#    for item in list(algorithm.V): print(item, '--------', algorithm.V[item])

start = timeit.default_timer()

testMDP()

stop = timeit.default_timer()
print('Time: ', stop - start)
