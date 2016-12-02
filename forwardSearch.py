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


class ForwardSearch:


	def __init__(self,game,d=3,gamma=0.7):
		self.game = game
		self.d = d
		self.gamma = gamma

	def getVisionLimitedBestAction(self,state):
		print (state)
		return self.helper(state,self.d)[0]

	def helper(self,state,d):
		if int(d)==0:
			return (None,0)

		a_best,v_best = None, float('-inf')
		drone_x, drone_y = state[0]
		target_x, target_y = state[1]
		for action in self.game.getAction(state):
			v = 0
			successors = self.game.getSuccessors(state,action)
			for successor in successors:
				nextState, reward, probability = successor
				a_next,v_next = self.helper(nextState,d-1)
				v = v + reward + self.gamma*probability*v_next
				# print ('  d=',d,' action=',action,' state=',nextState,' reward:',reward,' v:',v)

			drone_nx, drone_ny = nextState[0]
			target_nx, target_ny = nextState[1]
			diff = (abs(drone_x-target_x)+abs(drone_y-target_y))-\
				(abs(drone_nx-target_x)+abs(drone_ny-target_y))
			
			# if d==self.d:print (action,v,diff)

			if (abs(drone_x-target_x)+abs(drone_y-target_y))>4 and diff>0 and d==self.d:
				v += 500
			# if d==self.d:print (action,(abs(drone_x-target_x)+abs(drone_y-target_y)),v)
			if v > v_best:
				a_best = action
				v_best = v
		# if d==self.d:print ('best action:',a_best)
		return a_best,v_best


print ('------------ FORWARD SEARCH (vision limited) ------------\n')

iters = 100
scoreLs = []
timeLs = []

for t in range(100):
	startTime = time.time()
	game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt')
	FS = ForwardSearch(game,2,0.7)
	for i in range(1,iters+1):
		print ('iter ',t+1,":", i)
		game.drawCanvas(i,iters)

		state = game.getState()
		# print ('state: \n', state)

		# possibleActions = game.getAction(state)
		# print ('possible actions: \n', possibleActions) 

		# AI action
		action = FS.getVisionLimitedBestAction(game.getState())
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








