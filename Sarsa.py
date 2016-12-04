from game import Game
import pygame
import time
from collections import defaultdict
import time
import sys
import random
import math
from copy import copy
################################################################################################
################## Q Learning  #################################################################
################################################################################################
# Performs Q-learning.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearning(object):
    def __init__(self, getActions, discount, featureExtractor, explorationProb=0.2):
        self.actions = getActions #function
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

    def incorporateFeedback(self, state, action, reward, newState, newAction):
        if (newState == None):
            return None
        maxQ = self.getQ(newState, newAction)
        residual = reward + self.discount * maxQ - self.getQ(state, action)
        etaTimesR = residual * self.getStepSize()
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += etaTimesR * v


def identityFeatureExtractor(state, action):
    featureKey = (tuple(state), action)
    featureValue = 1
    return [(featureKey, featureValue)]


def decrease(d1, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * (-1)

def simulate(game, rl, numTrials=1, maxIterations=1000, drawCanvas=False,
             sort=False):

    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)


    totalRewards = []  # The rewards we get on each trial

    for trial in range(numTrials):
	# start game
        if drawCanvas is True:
           game.drawCanvas(0,maxIterations)
        state = game.getState()
	# record sequence, total discount, total reward
        sequence = [state]
        totalDiscount = 1
        totalReward = 0

        for i in range(maxIterations):
	    # drawCanvas
            if drawCanvas is True:
               time.sleep(0.1)
               game.drawCanvas(i,maxIterations)

	    # get action from Q-learning(explore OR exploit)
            action = rl.getAction(state)

	
	    # move in game
            reward = game.moveAction(action)

	    # New-State
            newState = game.getState()

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

	    # Q-learning Algorithm
            newAction = rl.getAction(newState)

            rl.incorporateFeedback(state, action, reward, newState, newAction)

	    # Accumulate the reward
            totalReward +=  reward

	    # update total discount
            totalDiscount *= rl.discount

	    # update state
            state = newState
    return game.score

######################################################################################################
game = Game(10,10,20,20,30,'obstaclesMap2.txt')
StepLimitForGame = 100 #100
StepLmitForTrain = 500 #500
featureExtractor = identityFeatureExtractor
print('Not Yet Train')
qlearning = QLearning(game.getAction, 0.9, featureExtractor, explorationProb=0.2)
NotTrainScore = simulate(game, qlearning, numTrials=1, maxIterations=StepLimitForGame, drawCanvas=False, sort=False)
print('Not Yet Train Score = ', NotTrainScore)

######################################################################################################
print('Training...')
NumOfTrials = 5000
start = time.time()
print('[')
for i in range(NumOfTrials):
        game = Game(10,10,20,20,30,'obstaclesMap2.txt')
        qlearning.explorationProb = (NumOfTrials-i)/NumOfTrials
        score = simulate(game, qlearning, numTrials=1, maxIterations=StepLmitForTrain, drawCanvas=False, sort=False)
         
        #print('trial =', i,'/',NumOfTrials,' exploreProb = ', qlearning.explorationProb, 'score =', score) 
        print(i,  NumOfTrials, qlearning.explorationProb, score, ';')

print(']')
end = time.time()
print('Times(s) =',end - start)
######################################################################################################
print('Final Test Trial...')

scores = []
times = []
for i in range(10):
	start = time.time()
	game = Game(10,10,20,20,30,'obstaclesMap2.txt')
	qlearning.explorationProb = 0.001
	(score) = simulate(game, qlearning, numTrials=1, maxIterations=StepLimitForGame, drawCanvas=False, sort=False)
	end = time.time()
	#print('Time', end - start,'Train Score = ', score)

	print(end - start, score)
	scores.append(score)
	times.append(end - start)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
print('Not Yet Train Score = ', NotTrainScore)
print('Train Score = ', mean(scores), 'time(s)',mean(times))
######################################################################################################



