from drone import Drone
from target import Target
from obstacle import Obstacle
from game import Game
import pygame
import numpy as np
import random
import math
import collections
import time

# Model Based Monte Carlo
# requires data to estimate transitions and rewards

# Model Free Monte Carlo
# requires data to find optimal Q

# policy: uniform random
# At each state, need to go through each possible random action
# Goal: Estimate Q_opt at state, and action

# Needs a dictionary of (s, a): (rewards, num of updates starting at 0)



class MonteCarlo:

	gamma = 0.7

	def __init__(self,game):
		self.game = game
		self.policy = {}
		self.x = game.GRIDS_X
		self.y = game.GRIDS_Y
		self.numIters = 10
		self.q_dict = {}

	def training(self, numOfMoves):
		startState = self.game.getState()
		startStateKey = tuple(startState)
		self.getData(startState, numOfMoves)


	def getData(self, state, numOfMoves):
		if numOfMoves > 0:
			stateKey = tuple(state)
			#print ("state:", state)
			possActions = self.game.getAction(state)

			action = random.choice(possActions)
			#print ("action", action)

			reward = self.game.moveAction(action)
			dictKey = (stateKey, action)
			if dictKey in self.q_dict:
				currReward, currNumUpdates = self.q_dict.get(dictKey)
				eta = 1.0/(1+currNumUpdates)
				newReward = (1-eta)*currReward + eta * reward
				self.q_dict[dictKey] = ( newReward , currNumUpdates+1 )
			else:
				self.q_dict[dictKey] = (reward, 1)
			#print ("new state:", self.game.getState())
			self.getData(self.game.getState(), numOfMoves-1)


	def getBestAction(self):
		state = self.game.getState()
		possActions = self.game.getAction(state)

		stateKey = tuple(state)
		maxVal = float('-inf')
		bestAction = None
		for action in possActions:
			dictKey = (stateKey, action)
			if dictKey in self.q_dict:
				reward, numUpdates = self.q_dict.get(dictKey)
				if reward > maxVal:
					maxVal = reward
					bestAction = action
		if bestAction == None:
			return random.choice(possActions)
		else:
			return bestAction


######################################################################################################
training = False
game = Game(10,10,20,20,30,'obstaclesMap2.txt')
MC = MonteCarlo(game)
######################################################################################################
if training:
	print('Training...')
	NumOfTrials = 5000
	start = time.time()
	for i in range(NumOfTrials):
		numOfMoves = 500
		game = Game(10,10,20,20,30,'obstaclesMap2.txt')
		MC.game = game
		MC.training(numOfMoves)
		if i %50 == 0:
			print ("Trial: ", i)
	end = time.time()
	print ("Training time:", end - start)
#print (MC.q_dict)
	np.save('trainedMonteCarlo.npy', MC.q_dict)
######################################################################################################
else:
	print("Testing...")
	MC.q_dict = np.load('trainedMonteCarlo.npy').item()
	#print(read_dictionary)

	numOfGames = 10
	gameStepLimit = 100
	totalScore = 0
	start = time.time()

	for j in range(numOfGames):
		game = Game(10,10,20,20,30,'obstaclesMap2.txt')
		MC.game = game
		for i in range(1,gameStepLimit+1):

			#print ('\niter: ', i)
			#game.drawCanvas(i,gameStepLimit)

			state = game.getState()
			#print ('state: \n', state)

			possibleActions = game.getAction(state)
			#print ('possible actions: \n', possibleActions)

			# AI action
			action = MC.getBestAction()
			#print ('take action: ',action)
			game.moveAction(action)
		totalScore += game.getScore()
	end = time.time()
	print ("Testing time:", (end - start) / numOfGames)
	averageGameScore = float(totalScore/numOfGames)
	print ("Average game score", averageGameScore)
