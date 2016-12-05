from drone import Drone
from target import Target
from obstacle import Obstacle
import pygame
from random import randint
import numpy as np
import time
import random
import collections

class Game:

	# Define some colors
	BLACK = (0, 0, 0)
	WHITE = (255, 255, 255)
	RED = (255, 0, 0)
	BLUE = (0,0,255)
	GREEN = (0,225,0)
	YELLOW = (225,225,0)
	WALL_SCORE = -20
	MARGIN = 1 # how long between each two squares

	offsetUp = 40 + randint(60,100)# offset for space to show score and time on the top of canvas
	offsetLeft = randint(60,100)
	offsetRight = randint(60,100)
	offsetDown = randint(60,100)

	moveLs = collections.Counter()
	# Initialize the setting of world
	def __init__(self,grids_x=10,grids_y=10,grid_width=20,grid_height=20,obstaclesPer=20,obstaclesMap=None):

		self.GRIDS_X = grids_x # how many squares in x direction
		self.GRIDS_Y = grids_y # how many squares in y direction
		self.WIDTH = grid_width # square width
		self.HEIGHT = grid_height # square height
		self.OBSTACLSNUM = int(float(grids_x)*grids_y*obstaclesPer/100)# how many obstacles in the setting
		self.grid = [ [0 for x in range(grids_x)] for y in range(grids_y) ]
		self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.score = 0
		self.drone = None
		self.target = None
		self.obstaclesMap = obstaclesMap
		self.obstaclesLs = []

		self.generateStartPoint()
		self.updateGrid()
		pygame.init()
		clock = pygame.time.Clock()
		clock.tick(60)
		self.screen = self.canvas()
		self.default_font = pygame.font.Font(None, 28)

		self.moveLs = collections.Counter()

	def generateObstacles(self):
		self.obstacleSet = set()
		# generate obstacles
		if self.obstaclesMap is not None:
			map_ob = open(self.obstaclesMap, 'r')
			for line in map_ob:
				loc = line.split(',')
				x,y = int(loc[0]),int(loc[1])
				self.obstacleSet.add( (x,y) )
		else:
			while len(self.obstacleSet) < self.OBSTACLSNUM:
				self.obstacleSet.add( (randint(0,self.GRIDS_X-1),randint(0,self.GRIDS_Y-1)) )

		for ob in self.obstacleSet:
			x,y = ob[0],ob[1]
			block = Obstacle(x,y)
			self.obstaclesLs.append(block)
			self.grid[y][x] = block

	def generateDrone(self):
		while True:
			x,y=randint(0,self.GRIDS_X-1),randint(0,self.GRIDS_Y-1)
			if self.grid[y][x] == 0:
				self.drone = Drone(x,y)
				self.grid[y][x] = self.drone
				break

	def generateTarget(self):
		while True:
			x,y=randint(2,self.GRIDS_X-3),randint(2,self.GRIDS_Y-3)
			if self.grid[y][x] == 0:
				self.target = Target(x,y,True)
				self.grid[y][x] = self.target
				break

	def copmuteObstacleReward(self):
		for obstacle in self.obstaclesLs:
			self.gridValue[obstacle.y][obstacle.x] += obstacle.reward


	def computeTargetReward(self,gridValue):
		target = self.target
		rewardRange = target.rewardRange
		x, y = self.target.x, self.target.y
		startX, endX = x - rewardRange, x + rewardRange
		startY, endY = y - rewardRange, y + rewardRange
		# target will compute correspond reward matrix
		target.computeRewardMatrix()
		rewardMatrix = target.rewardMatrix[1]

		for row_grid,row_reward in zip(range(startY,endY+1),range(2*rewardRange+1)):
			if row_grid < 0 or row_grid >=self.GRIDS_Y:	continue
			for col_grid, col_reward in zip(range(startX,endX+1),range(2*rewardRange+1)):
				if col_grid < 0 or col_grid >=self.GRIDS_X:	continue
				gridValue[row_grid][col_grid] += rewardMatrix[row_reward][col_reward]

	def computeValue(self):
		# calculate obstacle value
		self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.copmuteObstacleReward()
		self.computeTargetReward(self.gridValue)

	# randomly generate strat point on the grid for every stuff
	def generateStartPoint(self):
		self.generateObstacles()
		self.generateDrone()
		self.generateTarget()
		self.computeValue()


	def drawSquares(self,screen):
		white = np.asarray(list(self.WHITE))
		yellow = np.asarray(list(self.YELLOW))
		gradient = np.subtract(yellow,white)
		for row in range(self.GRIDS_Y):
 			for col in range(self.GRIDS_X):
 				# print (row,col,self.grid[row][col])
 				DistLeft = (self.MARGIN + self.WIDTH) * col + self.MARGIN
 				DistUp = (self.MARGIN + self.HEIGHT) * row + self.MARGIN
 				name = 'empty' if not self.grid[row][col] else self.grid[row][col].name
 				if not self.drone.checkExploration(col,row):
 					color = self.BLACK
 					if name is 'target':
	 					color = self.GREEN
 				else:
 					if self.grid[row][col] == 0:
 						# print (row,col,self.grid[row][col])
 						color = self.WHITE
 						if self.target.isReward(col,row):

		 					w = self.target.getReward(col,row)/self.target.maxReward
		 					# print (self.target.getReward(col,row),self.target.maxReward)
		 					color = tuple(np.add(white,np.multiply(gradient,w)))
 					else:
 						if name is 'drone':
 							color = self.BLUE
 						elif name is 'target':
 							color = self.GREEN
 						else:
 							color = self.RED
 				pygame.draw.rect(self.screen,color,[DistLeft+self.offsetLeft,DistUp+self.offsetUp,self.WIDTH,self.HEIGHT])


	def draw_text(self, text, font, surface, x, y, main_color, background_color=None):
 		textobj = font.render(text, True, main_color, background_color)
 		textrect = textobj.get_rect()
 		textrect.centerx = x
 		textrect.centery = y
 		surface.blit(textobj, textrect)

	def canvas(self):
		windowSize = [ (self.GRIDS_X+self.MARGIN)*self.WIDTH+self.offsetLeft+self.offsetRight, (self.GRIDS_Y+self.MARGIN)*self.HEIGHT+self.offsetUp+self.offsetDown]
		screen = pygame.display.set_mode(windowSize)
		pygame.display.set_caption("Awesome drone")
		# self.generateStartPoint()
		return screen

	def drawScore(self,screen,default_font,elapsed,timeEnd):
		self.drawSquares(self.screen)
		self.draw_text('points: {}'.format(self.score), default_font, self.screen,
              (self.GRIDS_X+self.MARGIN)*self.WIDTH*1/5, 20, self.GREEN)
		self.draw_text('time left: {}'.format(int(timeEnd-elapsed)), default_font, self.screen,
              (self.GRIDS_X+self.MARGIN)*self.WIDTH*4/5 , 20, self.GREEN)

	#after every stuff move, update the information of grid and value of grid
	def updateGrid(self):
		# remove original location
		target, drone = self.target, self.drone
		# undo last move
		# print (self.grid)
		self.grid[target.lastY][target.lastX] = 0
		self.grid[drone.lastY][drone.lastX] = 0

		# rebuild obstacles
		for obstacle in self.obstaclesLs:
			# self.grid[obstacle.lastY][obstacle.lastX] = 0
			self.grid[obstacle.y][obstacle.x] = obstacle

		# print (self.grid,'\n\n')
		self.grid[target.y][target.x] = target # for Target
		self.grid[drone.y][drone.x] = drone # for drone
		self.computeValue() # compue grid value
		self.score += self.gridValue[self.drone.y][self.drone.x]

	def hitWall(self):
		x, y = self.drone.x, self.drone.y
		if x<0 or y<0 or x>self.GRIDS_X-1 or y>self.GRIDS_Y-1:
			return True
		else:
			return False
	'''
	decide the preference move for target
	'''
	def targetMove(self):
		a = self.target.move(self.GRIDS_X,self.GRIDS_Y)
		# a = self.target.preferenceMove(self.GRIDS_X,self.GRIDS_Y)
		self.moveLs[a] += 1
		self.updateGrid()


	def eventHandler(self):
		for event in pygame.event.get():
			# If user clicked close
			if event.type == pygame.QUIT:
				return True
			if event.type == pygame.KEYDOWN:
				self.drone.controlMove(event)
				if self.hitWall():
					self.drone.undo()
					self.score += self.WALL_SCORE
				self.targetMove()
			self.updateGrid()
		return False

	def eventWait(self):
		# If user clicked close
		event = pygame.event.wait()

		if event.type == pygame.KEYDOWN:
			self.drone.controlMove(event)
			if self.hitWall():
				self.drone.undo()
				self.score += self.WALL_SCORE
			self.targetMove()
			self.updateGrid()
			return False
		else:
			self.updateGrid()
			return True

	def drawCanvas(self,i,iters):
		self.screen.fill(self.BLACK)
		self.drawScore(self.screen,self.default_font,i,iters)
		pygame.display.flip()

	# get every possible next state and reward
	def getVisionLimitedNextStateAndReward(self,state,action):
		self.drone.backUp()
		self.drone.x, self.drone.y = state[0]
		self.drone.AImove(action)
		droneState = (self.drone.x, self.drone.y)
		nextStates = []
		rewards = []

		for act in self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y):
			self.target.AImove(act)
			self.computeValue()
			reward = self.gridValue[self.drone.y][self.drone.x]

			if (self.drone.x,self.drone.y) not in self.drone.backExp and \
				(self.drone.x,self.drone.y) in self.obstacleSet:
				reward += 50

			state = [droneState, (self.target.x,self.target.y), (self.target.face, self.target.faceAngle)]
			nextStates.append(state)
			rewards.append(reward)
			self.target.undo()
		# self.drone.undo()
		self.drone.recoverBackUp()
		return nextStates,rewards

	def getVisionLimitedSuccessors(self,state,action):
		self.drone.backUp()
		self.drone.x, self.drone.y = state[0]
		self.drone.AImove(action)
		droneState = (self.drone.x, self.drone.y)
		successors = []
		nextStates = []
		rewards = []

		for act in self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y):
			self.target.AImove(act)
			self.computeValue()
			reward = self.gridValue[self.drone.y][self.drone.x]

			if (self.drone.x,self.drone.y) not in self.drone.backExp and \
				(self.drone.x,self.drone.y) in self.obstacleSet:
				reward += 50

			nextState = [droneState, (self.target.x,self.target.y), (self.target.face, self.target.faceAngle)]
			nextStates.append(nextState)
			probability = self.getPossibility(state,action,nextState)
			successor = [nextState, reward, probability]
			successors.append(successor)
			rewards.append(reward)
			self.target.undo()
		# self.drone.undo()
		self.drone.recoverBackUp()
		return successors




	def faceApproximation(self,state):
		targetPos = state[0]
		res = 0
		if abs(targetPos[0])<=3 and abs(targetPos[1])<=3:
			res = sum(state[1])
		return res

	# local approximation of relative loaction of target from (-n~n,-m~m) to (-10~10,-10~10)
	def targetApproximation(self,targetPos):
		# maintain exactly same relative position within (-5~5,-5~5), it is close to drone which is important
		# goal is to convert (-originalXRange~originalXRange,-originalYRange~originalYRange) to (-10~10,-10~10)
		originalXRange = self.GRIDS_X
		originalYRange = self.GRIDS_Y
		# local approximation for x
		if abs(targetPos[0])<=5:
			newX = targetPos[0]
		else:
			newX = 6 if targetPos[0]>0 else -6
		# local approximation for y
		if abs(targetPos[1])<=5:
			newY = targetPos[1]
		else:
			newY = 6 if targetPos[1]>0 else -6
		return (round(newX),round(newY))

	# local approximate of location of obstacles from 5*5 to 3*3
	def obstacleApproximation(self,obstaclePos):
		# local approximation for x
		if obstaclePos[0]<0:
			newX = -1
		elif obstaclePos[0]==0:
			newX = 0
		else:
			newX = 1

		# local approximation for y
		if obstaclePos[1]<0:
			newY = -1
		elif obstaclePos[1]==0:
			newY = 0
		else:
			newY = 1

		return(newX,newY)


	##################### SARS structure #####################
	# return state with location of drone and target [ droneLocation, targetLocatoin, face and faceangle ]
	def getState(self):
		return [(self.drone.x,self.drone.y),(self.target.x,self.target.y) ,( self.target.face, self.target.faceAngle)]

	# state become [relative position from drone to target, target's face situation, relative position to obstacles within 2 block distance]
	def getState2(self):
		state = []
		# relative position from drone to target
		state.append((self.target.x-self.drone.x,self.target.y-self.drone.y))
		# target's face situation
		state.append((self.target.face, self.target.faceAngle))
		# relative position to obstacles within 2 block distance
		for obstacle in self.obstaclesLs:
			if abs(obstacle.y-self.drone.y)<=2 and abs(obstacle.x-self.drone.x)<=2:
				state.append((obstacle.x-self.drone.x,obstacle.y-self.drone.y))
		return state

	def getLocalApporximationState(self):
		state = self.getState2()
		newState = []
		newState.append(self.targetApproximation(state[0]))
		newState.append(self.faceApproximation(state))
		for i in range(2,len(state)):
			newPos = self.obstacleApproximation(state[i])
			if newPos not in newState:
				newState.append(newPos)

		return newState

	# return possible next action
	def getAction(self,state):
		self.drone.backUp()
		self.drone.x, self.drone.y = state[0]
		actions = self.drone.possibleActions(self.GRIDS_X,self.GRIDS_Y)
		self.drone.recoverBackUp()

		return actions
	# drone's move is for sure. assume target move randomly
	def getPossibility(self,state,action,nextState):
		return 1.0/len(self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y))

	# return [ nextState, reward, probability]
	def getSuccessors(self,state,action):
		self.drone.backUp()
		self.drone.x, self.drone.y = state[0]
		self.target.backUp()
		self.target.x, self.target.y = state[1]
		self.target.face, self.target.faceAngle= state[2]

		self.drone.AImove(action)
		droneState = (self.drone.x, self.drone.y)
		successors = []
		nextStates = []
		rewards = []
		for act in self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y):
			self.target.AImove(act)
			self.computeValue()
			reward = self.gridValue[self.drone.y][self.drone.x]
			nextState = [droneState, (self.target.x,self.target.y), (self.target.face, self.target.faceAngle)]
			probability = self.getPossibility(state,action,nextState)
			successor = [nextState, reward, probability]
			successors.append(successor)
			self.target.undo()
		# self.drone.undo()
		self.drone.recoverBackUp()
		self.target.recoverBackUp()
		return successors


	# get every possible next state and reward
	def getNextStateAndReward(self,state,action):
		self.drone.backUp()
		self.drone.x, self.drone.y = state[0]
		self.drone.AImove(action)
		droneState = (self.drone.x, self.drone.y)
		nextStates = []
		rewards = []
		for act in self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y):
			self.target.AImove(act)
			self.computeValue()
			reward = self.gridValue[self.drone.y][self.drone.x]
			state = [droneState, (self.target.x,self.target.y), (self.target.face, self.target.faceAngle)]
			nextStates.append(state)
			rewards.append(reward)
			self.target.undo()
		# self.drone.undo()
		self.drone.recoverBackUp()
		return nextStates,rewards

	# move the drone and return the reward you get
	def moveAction(self,action):
		self.drone.AImove(action)
		self.targetMove()
		self.updateGrid()
		return self.gridValue[self.drone.y][self.drone.x]

	# return the reward at given state at current time
	def getReward(self,state):
		loc = state[0]
		return self.gridValue[loc[1]][loc[0]]

	# return gridValue which store all rewards at current time
	def getGrid(self):
		return self.gridValue

	# return the region that drone already explored
	def getExploredArea(self):
		return self.drone.exploredSet

	# return the cumulative score
	def getScore(self):
		return self.score

	def getObstaclesAronud(self):
		xStart = 0 if self.drone.x-2<0 else self.drone.x-2
		xEnd = self.GRIDS_X-1 if self.drone.x+2>self.GRIDS_X-1 else self.drone.x+2
		yStart = 0 if self.drone.y-2<0 else self.drone.y-2
		yEnd = self.GRIDS_Y-1 if self.drone.y+2>self.GRIDS_Y-1 else self.drone.y+2
		obstaclesAround = []
		#print (xStart,xEnd,yStart,yEnd)
		for x in range(xStart,xEnd+1):
			for y in range(yStart,yEnd+1):
				name = 'empty' if not self.grid[y][x] else self.grid[y][x].name
				if name == 'obstacle':
					distance = abs(x-self.drone.x)+abs(y-self.drone.y)
					obstaclesAround.append([(x-self.drone.x,y-self.drone.y),distance])
		return obstaclesAround

	# return reward matrix with assigned droneVal which indicate the location of drone
	def getRewardMatrix(self,droneVal):
		res = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.computeTargetReward(res)
		for obstacle in self.obstaclesLs:
			if abs(obstacle.y-self.drone.y)<=2 and abs(obstacle.x-self.drone.x)<=2:
				res[obstacle.y][obstacle.x] += obstacle.reward
		res[self.drone.y][self.drone.x] = droneVal
		return res



	##################### SARS structure End#####################
