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

class MonteCarloEval:

	def __init__(self,game):
		self.game = game
		self.Qsa = collections.defaultdict(int)
		self.policy = {}
		self.N = collections.defaultdict(int) # to count how many times the state was evaluated

	# def stateLocalApproximate(self,state):
	# 	newState = []
	# 	newState.append(self.targetApproximation(state[0]))
	# 	newState.append(self.faceApproximation(state))
	# 	for i in range(2,len(state)):
	# 		newPos = self.obstacleApproximation(state[i])
	# 		if newPos not in newState:
	# 			newState.append(newPos)

	# 	return tuple(newState)

	# def faceApproximation(self,state):
	# 	targetPos = state[0]
	# 	res = 0
	# 	if abs(targetPos[0])<=3 and abs(targetPos[1])<=3:
	# 		res = sum(state[1])
	# 	return res

	# # local approximation of relative loaction of target from (-n~n,-m~m) to (-10~10,-10~10)
	# def targetApproximation(self,targetPos):
	# 	# maintain exactly same relative position within (-5~5,-5~5), it is close to drone which is important
	# 	# goal is to convert (-originalXRange~originalXRange,-originalYRange~originalYRange) to (-10~10,-10~10)
	# 	originalXRange = self.game.GRIDS_X
	# 	originalYRange = self.game.GRIDS_Y
	# 	# local approximation for x
	# 	if abs(targetPos[0])<=3:
	# 		newX = targetPos[0]
	# 	else:
	# 		newX = targetPos[0]*6/(originalXRange+1)
	# 	# local approximation for y
	# 	if abs(targetPos[1])<=3:
	# 		newY = targetPos[1]
	# 	else:
	# 		newY = targetPos[1]*6/(originalYRange+1)
	# 	return (round(newX),round(newY))
	
	# # local approximate of location of obstacles from 5*5 to 3*3
	# def obstacleApproximation(self,obstaclePos):
	# 	# local approximation for x
	# 	if obstaclePos[0]<0:
	# 		newX = -1
	# 	elif obstaclePos[0]==0:
	# 		newX = 0
	# 	else:
	# 		newX = 1

	# 	# local approximation for y
	# 	if obstaclePos[1]<0:
	# 		newY = -1
	# 	elif obstaclePos[1]==0:
	# 		newY = 0
	# 	else:
	# 		newY = 1

	# 	return(newX,newY)

	def getBestAction(self,state):
		state2 = tuple(game.getLocalApporximationState())
		maxVal = float('-inf')
		bestAction = None
		for action in game.getAction(state):
			sa = tuple((state2,action))
			if self.Qsa[sa]>maxVal:
				bestAction = action
				maxVal = self.Qsa[sa]
		return bestAction

	def chooseAction(self,state,state2,actions):
		minVal = float('inf')
		res = None
		for action in actions:
			sa = tuple((state2,action))
			if sa not in self.N.keys():
				return action
			else:
				if self.N[sa]<minVal:
					minVal = self.N[sa]
					res = action
		return res

	def loadQsa(self):
		self.Qsa = np.load('Qsa.npy').item()
		self.N = np.load('N.npy').item()
		for sa in self.Qsa.keys():
			print (sa,'  Q: ',self.Qsa[sa], '; N: ',self.N[sa])

	def training(self,times,iters):
		for t in range(times):
			print ('times:',t,len(self.Qsa))
			history = []
			game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt') # BE CAREFUL!!!!!  training game setting should be same as real game
			
			for i in range(1,iters+1):
				state = game.getState()
				# state2 = game.getState2()
				# state2 = self.stateLocalApproximate(state2)
				state2 = tuple(game.getLocalApporximationState())

				action = self.chooseAction(state,state2,game.getAction(state))
				self.N[tuple((state2,action))]+=1
				history.append(tuple((state2,action)))
				game.moveAction(action)

			score = game.getScore()
			for sa in history:
				self.Qsa[sa] += 1./self.N[sa]*score
		# Save
		np.save('Qsa.npy', self.Qsa) 
		np.save('N.npy', self.N) 


			

training = True
trainingTime = 30000
itersPerTime = 100
testingTime = 100
# gameSetting = 'Game(10,10,20,20,20,\'./environment/obstaclesMap2.txt\')'
game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt')
MCE = MonteCarloEval(game)

if training:
	print ('------------ MonteCarloEval SEARCH Trainning ------------\n')
	# MCE.loadQsa()
	MCE.training(trainingTime,itersPerTime)
else:
	print ('------------ MonteCarloEval SEARCH ------------\n')
	scoreLs = []
	timeLs = []
	MCE.loadQsa()
	for t in range(testingTime):
		startTime = time.time()
		game = Game(10,10,20,20,20,'./environment/obstaclesMap2.txt')
		for i in range(1,itersPerTime+1):
			print ('iter ',t+1,":", i)
			game.drawCanvas(i,itersPerTime)
			state = game.getState()
			
			# AI action
			action = MCE.getBestAction(state)
			print ('action: ',action)
			pause = input("press enter to next step")
			game.moveAction(action)

		elapsedTime = time.time() - startTime
		print ('score: ',game.getScore(),'; time: ',elapsedTime ,'s')
		scoreLs.append(game.getScore())
		timeLs.append(elapsedTime)

	print (scoreLs)
	print (timeLs)



