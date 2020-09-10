'''
file: util.py
authors: Kento Perera, Timothy Sah, and Dean Stratakos
date: December 1, 2019
----------
This file contains helper methods for our CS 221 Project.
We borrowed code structure from blackjack/util.py.
'''

import math
import os.path
import collections, random
import csv

'''
class: MDPAlgorithm
----------
An algorithm that solves an MDP (i.e., computes the optimal policy).
'''
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")

'''
class: ValueIteration
----------
Implementation of the Value Iteration algorithm.
'''
class ValueIteration(MDPAlgorithm):
    '''
    function: solve
    ----------
    Solves the MDP using value iteration. This function sets
        - self.V to the dictionary mapping states to optimal values
        - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: value iteration stops when all of the
    values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            if state == (): return 0
            return sum(prob * (reward + mdp.discount() * V[newState[0]]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state[0]] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state[0]] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            V = newV
            if max(abs(V[state[0]] - newV[state[0]]) for state in mdp.states) < epsilon: break

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print(("ValueIteration: %d iterations" % numIters))
        self.pi = pi
        self.V = V

'''
class: MDP
----------
An abstract class representing a Markov Decision Process (MDP).
'''
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
                        if len(self.states) % 1000 == 0: print(len(self.states))
        print ("%d states" % len(self.states))

'''
class: RLAlgorithm
----------
Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
to know is the set of available actions to take.  The simulator (see
simulate()) will call getAction() to get an action, perform the action, and
then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
its parameters.
'''
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

'''
function: simulate
----------
Perform |numTrials| of the following:
On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulate the
RL algorithm according to the dynamics of the MDP.
Each trial will run for at most |maxIterations|.
Return the list of rewards that we get for each trial.
'''
def simulate(mdp, rl, numTrials=10, maxIterations=100, verbose=True, sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    # Clear the memory log (due to some error)
    # f = open('qStarMemoryLog.csv', 'w')
    # f.truncate()    
    # f.close()

    # Read in past qStar data from qStar memory log
    f = open('qStarMemoryLog.csv', 'r')
    fileReader = csv.reader(f)
    for n, line in enumerate(fileReader):
        if n == 0: continue
        stateTup = []
        stateStr = line[0].split()
        for i in range(0, len(stateStr), 2):
            stateStr[i] = str(stateStr[i].strip('()\','))
            stateStr[i+1] = int(stateStr[i+1].strip('()\','))
            stateTup.append((stateStr[i], stateStr[i+1]))
        actionStr = line[1].split()
        actionTup = []
        for i in range(0, len(actionStr), 2):
            actionStr[i] = str(actionStr[i].strip('()\','))
            actionStr[i+1] = int(actionStr[i+1].strip('()\','))
            actionTup.append((actionStr[i], actionStr[i+1]))
        state, action, reward = tuple(stateTup), tuple(actionTup), float(line[2])
        rl.qStarActions[state] = [action, reward]
    f.close()
    print('Before: ', len(rl.qStarActions))

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        print('Trial number: ', trial)
        state = mdp.startState()
        sequence = [state]
        sarSequence = []
        totalDiscount = 1
        totalReward = 0
        for i in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action) # returns one transition because there is only one successor state
            if sort:
                transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break
            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)
            sarSequence.append((state, action, reward))

            rl.incorporateFeedback(state, action, reward, newState)
            if state[0] not in rl.qStarActions.keys() or reward > rl.qStarActions[state[0]][1]:
                rl.qStarActions[state[0]] = [action, reward]
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print(("Trial %d (totalReward = %s): %s\n" % (trial, totalReward, sequence)))
        for s, a, r in sarSequence:
            list(s[0]).sort(key = lambda x: x[0]+str(x[1]))
            list(a).sort(key = lambda x: x[0]+str(x[1]))
            if rl.qStarActions[tuple(s)[0]] == [] or r >= rl.qStarActions[tuple(s)[0]][1]:
                rl.qStarActions[tuple(s)[0]] = [a, r]
        totalRewards.append(totalReward)

        print('After: ', len(rl.qStarActions))
        # Write in new qStar data to qStar memory log
        f = open('qStarMemoryLog.csv', 'w')
        fileWriter = csv.writer(f)
        fileWriter.writerow(['State', 'Action', 'Reward'])
        for key in rl.qStarActions:
            fileWriter.writerow([str(key), str(rl.qStarActions[key][0]), str(rl.qStarActions[key][1])])
        f.close()
        # 

    return totalRewards
