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

# I assumed that the drone already knew all the environment setting of game
# so, I do value iteration every time(because the target can move) before doing next action
# state: location of drone
# action: east, west, north, south

# because the drone already knew the setting of the game and value iteration give the 
# optimal solution. This method should be our upper limit




class ValueIteratoin:

	

	def __init__(self,game,convergeError,asynchronous=True,gamma=0.7):
		self.game = game
		self.Us = collections.defaultdict(int)
		self.nextUs = collections.defaultdict(int)
		self.policy = {}
		self.asynchronous = asynchronous
		self.x = game.GRIDS_X
		self.y = game.GRIDS_Y
		self.gamma = gamma
		self.convergeError = float(convergeError)*(1.-self.gamma)/self.gamma

	def getBestAction(self):
		state = game.getState()
		self.computeValue()
		return self.policy[state[0]]

	def computeValue(self):
		error = float('inf')
		while error>self.convergeError:
			error = 0
			for x in range(self.x):
				for y in range(self.y):
					state = (x,y)
					newVal = self.findMaxAction(state)
					error = max(error, abs(newVal - self.Us[state]) )
					self.Us[state] = newVal
			# error = math.sqrt(error)
			# print ('error: ',error)
		# for state,action in self.policy.items():
		# 	print ('state: ',state,'; actoin: ',action,'; value',self.Us[state])

	def findMaxAction(self,state):
		s = state
		state = tuple([state,None,None])
		actions = game.getAction(state)
		maxVal = float('-inf')
		# loop all possible actions the drone can do at this state
		for action in actions:
			actionVal = 0
			# find the reward and nextState when dron take this action
			nextStates,rewards = game.getNextStateAndReward(state,action)

			for nextState,reward in zip(nextStates,rewards):
				# get probability of getting the reward and next state
				prob = game.getPossibility(state,action,nextState)
				actionVal += (reward)*prob + self.gamma*prob*self.Us[nextState[0]]

			if actionVal > maxVal:
				maxVal = actionVal
				self.policy[s] = action
		return maxVal

	def getVisionLimitedBestAction(self):
		state = game.getState()
		self.computeVisionLimitedValue()
		return self.policy[state[0]]

	def computeVisionLimitedValue(self):
		error = float('inf')
		while error>self.convergeError:
			error = 0
			for x in range(self.x):
				for y in range(self.y):
					state = (x,y)
					newVal = self.findVisionLimitedMaxAction(state)
					error = max(error, abs(newVal - self.Us[state]) )

					if self.asynchronous:
						self.Us[state] = newVal
					else:
						self.nextUs[state] = newVal
			if not self.asynchronous:
				temp = self.Us
				self.Us = self.nextUs
				self.nextUs = temp

	def findVisionLimitedMaxAction(self,state):
		s = state
		state = tuple([state,None,None])
		actions = game.getAction(state)
		maxVal = float('-inf')
		# loop all possible actions the drone can do at this state
		for action in actions:
			actionVal = 0
			# find the reward and nextState when dron take this action
			nextStates,rewards = game.getVisionLimitedNextStateAndReward(state,action)

			for nextState,reward in zip(nextStates,rewards):
				# get probability of getting the reward and next state
				prob = game.getPossibility(state,action,nextState)
				actionVal += (reward)*prob + self.gamma*prob*self.Us[nextState[0]]

			if actionVal > maxVal:
				maxVal = actionVal
				self.policy[s] = action
		return maxVal







# print ('------------ VALUE ITERATION ------------\n')
# game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt')
# VI = ValueIteratoin(game,0.001)

# iters = 50
# for i in range(1,iters+1):
# 	print ('\niter: ', i)
# 	game.drawCanvas(i,iters)

# 	state = game.getState()
# 	print ('state: \n', state)

# 	possibleActions = game.getAction(state)
# 	print ('possible actions: \n', possibleActions) 

# 	# AI action
# 	action = VI.getBestAction()
# 	print ('take action: ',action)
# 	game.moveAction(action)
# 	# pause = input("press enter to next step")
		
# print (game.getScore())




print ('------------ VALUE ITERATION (vision limited) ------------\n')

iters = 100
scoreLs = []
timeLs = []

for t in range(10):
	startTime = time.time()
	game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt')
	VI = ValueIteratoin(game,0.01,True,0.7)
	for i in range(1,iters+1):
		print ('iter ',t+1,":", i)
		game.drawCanvas(i,iters)

		state = game.getState()
		# print ('state: \n', state)

		possibleActions = game.getAction(state)
		# print ('possible actions: \n', possibleActions) 

		# AI action
		action = VI.getVisionLimitedBestAction()
		# print ('take action: ',action)
		game.moveAction(action)
		# print (game.gridValue)
		# pause = input("press enter to next step")
	elapsedTime = time.time() - startTime
	print ('score: ',game.getScore(),'; time: ',elapsedTime ,'s')
	scoreLs.append(game.getScore())
	timeLs.append(elapsedTime)

print (scoreLs)
print (timeLs)








