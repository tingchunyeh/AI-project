from drone import Drone
from target import Target
from obstacle import Obstacle
from simulator2 import Simulator2
import pygame
import numpy as np
import random
import math
import collections

# I assumed that the drone already knew all the environment setting of game
# so, I do value iteration every time(because the target can move) before doing next action
# state: location of drone
# action: east, west, north, south

# because the drone already knew the setting of the game and value iteration give the 
# optimal solution. This method should be our upper limit




class ValueIteratoin:

	gamma = 0.7

	def __init__(self,game,convergeError):
		self.game = game
		self.Us = collections.defaultdict(int)
		self.policy = {}
		self.x = game.GRIDS_X
		self.y = game.GRIDS_Y
		self.convergeError = convergeError

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
					error = (newVal - self.Us[state])**2
					self.Us[state] = newVal
			error = math.sqrt(error)
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



game = Simulator2(10,10,20,20,20,False)
VI = ValueIteratoin(game,0.001)

print ('------------ VALUE ITERATION ------------\n')


iters = 50

for i in range(iters):
	print ('\niter: ', i)
	
	game.drawCanvas(i,iters)

	state = game.getState()
	print ('state: \n', state)

	# exploredSet = game.getExploredArea()
	# print ('explored area: \n', exploredSet)

	# gridValue = game.getGrid()
	# print ('value of grid: \n', gridValue )

	possibleActions = game.getAction(state)
	print ('possible actions: \n', possibleActions) 


	# for action in possibleActions:
	# 			print ('	action:',action)
	# 			nextStates,rewards = game.getNextStateAndReward(state,action)
	# 			for nextState,reward in zip(nextStates,rewards):
	# 				probability = game.getPossibility(state,action,nextState)
	# 				print ('		nextState:',nextState,'; reward:',reward,'; probability: ', probability)
	
	# AI action
	action = VI.getBestAction()
	print ('take action: ',action)
	game.moveAction(action)
	# pause = input("press enter to next step")

	# # Human action
	# action = input(">>> next action: ")
	# print ('take action: ', action)
	# game.moveAction(action)
		
print (game.getScore())








