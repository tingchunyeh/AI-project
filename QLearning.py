from drone import Drone
from target import Target
from obstacle import Obstacle
from simulator2 import Simulator2
import pygame
from random import randint
import numpy as np
import time
import math
import random
from collections import defaultdict

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

    def incorporateFeedback(self, state, action, reward, newState):
        if (newState == None):
            return None
        maxQ = max(self.getQ(newState, newAction) for newAction in self.actions(newState))
        residual = reward + self.discount * maxQ - self.getQ(state, action)
        etaTimesR = residual * self.getStepSize()
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += etaTimesR * v


def identityFeatureExtractor(state, action):
    featureKey = (tuple(state), action)
    featureValue = 1
    return [(featureKey, featureValue)]



def simulate(game, rl, numTrials=1, maxIterations=1000, verbose=False,
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
        game.drawCanvas(0,maxIterations)
        state = game.getState()
	# record sequence, total discount, total reward
        sequence = [state]
        totalDiscount = 1
        totalReward = 0

        for i in range(maxIterations):
            print('state',i,state)
	    # drawCanvas
            game.drawCanvas(i,maxIterations)

	    # get action from Q-learning(explore OR exploit)
            action = rl.getAction(state)
	
	    # move in game
            game.moveAction(action)

	    # based on this action, what is next state/reward?
            newStates, rewards = game.getNextStateAndReward(state,action)

            # if there is no transistions, no reward, no newState
            if len(newStates) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

	    # New-State
            newState = game.getState()
            reward = 0
            for nextState,rewardd in zip(newStates,rewards):
                probability = game.getPossibility(state,action,nextState)
                if nextState is newState:
                   reward = rewardd

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

	    # Q-learning Algorithm
            rl.incorporateFeedback(state, action, reward, newState)

	    # Accumulate the reward
            totalReward += totalDiscount * reward

	    # update total discount
            totalDiscount *= rl.discount

	    # update state
            state = newState
        if verbose:
            print(trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards


######################################################################################################
game = Simulator2(10,10,20,20,20,False)
qlearning = QLearning(game.getAction, 0.9, identityFeatureExtractor, explorationProb=0.2)
######################################################################################################


simulate(game, qlearning, numTrials=10, maxIterations=1000, verbose=False, sort=False)





