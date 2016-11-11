from drone import Drone
from target import Target
from obstacle import Obstacle
import pygame
from random import randint
import numpy as np
import time
import random

class Simulator2:

	# Define some colors
	BLACK = (0, 0, 0)
	WHITE = (255, 255, 255)
	RED = (255, 0, 0)
	BLUE = (0,0,255)
	GREEN = (0,225,0)
	YELLOW = (225,225,0)
	
	WALL_SCORE = -20
	MARGIN = 1 # how long between each two squares
	score = 0
	drone = None
	target = None
	obstaclesLs = []
	offsetUp = 40 + randint(40,80)# offset for space to show score and time on the top of canvas
	offsetLeft = randint(40,80)
	offsetRight = randint(40,80)
	offsetDown = randint(40,80)
	timePeriod = 0.5 # Used to manage how fast the screen updates
	timeEnd = 30

	# Initialize the setting of world
	def __init__(self,grids_x=10,grids_y=10,grid_width=20,grid_height=20,obstaclesPer=20,simulate=False):
		self.GRIDS_X = grids_x # how many squares in x direction
		self.GRIDS_Y = grids_y # how many squares in y direction
		self.WIDTH = grid_width # square width
		self.HEIGHT = grid_height # square height
		self.OBSTACLSNUM = int(float(grids_x)*grids_y*obstaclesPer/100)# how many obstacles in the setting
		self.grid = [ [0 for x in range(grids_x)] for y in range(grids_y) ]
		self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.simulate = simulate
		self.iter = 0
		self.generateStartPoint()
		self.updateGrid()

	def generateObstacles(self):
		obstacleSet = set()
		# generate obstacles
		while len(obstacleSet) < self.OBSTACLSNUM:
			obstacleSet.add( (randint(0,self.GRIDS_X-1),randint(0,self.GRIDS_Y-1)) )
		for ob in obstacleSet:
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
				self.target = Target(x,y)
				self.grid[y][x] = self.target
				break

	def copmuteObstacleReward(self):
		for obstacle in self.obstaclesLs:
			self.gridValue[obstacle.y][obstacle.x] = obstacle.reward

		
	def computeTargetReward(self):
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
				self.gridValue[row_grid][col_grid] += rewardMatrix[row_reward][col_reward]

	def computeValue(self):
		# calculate obstacle value
		self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.copmuteObstacleReward()
		self.computeTargetReward()
		
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
 				DistLeft = (self.MARGIN + self.WIDTH) * col + self.MARGIN
 				DistUp = (self.MARGIN + self.HEIGHT) * row + self.MARGIN
 				name = 'empty' if not self.grid[row][col] else self.grid[row][col].name
 				if not self.drone.checkExploration(col,row):
 					color = self.BLACK
 					if name is 'target':
	 					color = self.GREEN
 				else:
 					if self.grid[row][col] == 0:
 						color = self.WHITE
 						if self.target.isReward(col,row):
		 					w = self.target.getReward(col,row)/self.target.maxReward
		 					color = tuple(np.add(white,np.multiply(gradient,w)))
 					else:
 						if name is 'drone':
 							color = self.BLUE
 						elif name is 'target':
 							color = self.GREEN
 						else:
 							color = self.RED
 				pygame.draw.rect(screen,color,[DistLeft+self.offsetLeft,DistUp+self.offsetUp,self.WIDTH,self.HEIGHT])


	def draw_text(self, text, font, surface, x, y, main_color, background_color=None):
 		textobj = font.render(text, True, main_color, background_color)
 		textrect = textobj.get_rect()
 		textrect.centerx = x
 		textrect.centery = y
 		surface.blit(textobj, textrect)

	def canvas(self):
		windowSize = [ (self.GRIDS_X+self.MARGIN)*self.WIDTH+self.offsetLeft+self.offsetRight, (self.GRIDS_Y+self.MARGIN)*self.HEIGHT+self.offsetUp+self.offsetDown]
		screen = pygame.display.set_mode(windowSize)
		# Set title of screen
		pygame.display.set_caption("Awesome drone")
		# set up obstacles, drone and target
		self.generateStartPoint()
		return screen

	def drawScore(self,screen,default_font,elapsed,timeEnd):
		self.drawSquares(screen)
		self.draw_text('points: {}'.format(self.score), default_font, screen,
              (self.GRIDS_X+self.MARGIN)*self.WIDTH*1/5, 20, self.GREEN)
		self.draw_text('time left: {}'.format(int(timeEnd-elapsed)), default_font, screen,
              (self.GRIDS_X+self.MARGIN)*self.WIDTH*4/5 , 20, self.GREEN)

	#after every stuff move, update the information of grid and value of grid
	def updateGrid(self):
		# remove original location
		target, drone = self.target, self.drone
		# undo last move
		self.grid[target.lastY][target.lastX] = 0
		self.grid[drone.lastY][drone.lastX] = 0

		# rebuild obstacles
		for obstacle in self.obstaclesLs:
			self.grid[obstacle.lastY][obstacle.lastX] = 0
			self.grid[obstacle.y][obstacle.x] = obstacle

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

	def targetMove(self):
		self.target.randmove(self.GRIDS_X,self.GRIDS_Y)
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


	def start(self):
		pygame.init()
		screen = self.canvas()
		default_font = pygame.font.Font(None, 28)

		start_time = time.time()
		lastMoveTime = start_time
		clock = pygame.time.Clock()
		
		# Loop until the user clicks the close button.
		done = False
		while not done:
			screen.fill(self.BLACK)
			elapsed = time.time() - start_time
			done = self.eventHandler()
			self.drawScore(screen,default_font,elapsed,self.timeEnd) # draw
			clock.tick(60) # Limit to 60 frames per second
			pygame.display.flip() # Go ahead and update the screen with what we've drawn.
			print (self.getState()[0])
			if elapsed > self.timeEnd:
				print (self.score)
				break
			print (self.gridValue,'\n')
		pygame.quit()


	def simulation(self):
		pygame.init()
		screen = self.canvas()
		default_font = pygame.font.Font(None, 28)

		start_time = time.time()
		lastMoveTime = start_time
		clock = pygame.time.Clock()
		

		# Loop until the user clicks the close button.
		done = False
		self.updateGrid()
		while not done:
			screen.fill(self.BLACK)
			elapsed = time.time() - start_time
			state = self.getState()
			actions = self.getAction(state)
			print ('state:',state)
			print ('actions:',actions)
			print (self.drone.x,self.drone.y)

			for action in actions:
				print ('	action:',action)
				nextStates,rewards = self.getNextStateAndReward(state,action)
				for nextState,reward in zip(nextStates,rewards):
					print ('		nextState:',nextState,'; reward:',reward)

			for event in pygame.event.get():
				# If user clicked close
				if event.type == pygame.QUIT:
					done = True

			print (self.drone.x,self.drone.y)
			self.drawScore(screen,default_font,elapsed,self.timeEnd) # draw
			clock.tick(60) # Limit to 60 frames per second
			pygame.display.flip() # Go ahead and update the screen with what we've drawn.
			nextAction =  input(">>> next action: ")
			self.drone.AImove(nextAction)
			print (self.drone.x,self.drone.y)
			self.targetMove()
			self.updateGrid()


			# print (self.gridValue,'\n')
		pygame.quit()
	



	##################### SARS structure #####################
	# return state with location of drone and target [ droneLocation, targetLocatoin, face and faceangle ]
	def getState(self):
		return [(self.drone.x,self.drone.y),(self.target.x,self.target.y) ,( self.target.face, self.target.faceAngle)]

	# get possible next action
	def getAction(self,state):
		self.drone.x, self.drone.y = state[0]
		return self.drone.possibleActions(self.GRIDS_X,self.GRIDS_Y)

	# drone' move is for sure. assume target move randomly
	def getPossibility(self,state,action,nextState):
		return 1.0/len(self.target.possibleActions(self.GRIDS_X,self.GRIDS_Y))

	# get every possible next state and reward
	def getNextStateAndReward(self,state,action):
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
		self.drone.undo()
		return nextStates,rewards

	def moveAction(self,action):
		print ('movemvoemvomeovmeovm')
		self.iter += 1
		self.drone.AImove(action)
		self.targetMove()
		self.updateGrid()

	def getGrid(self):
		return self.gridValue

	def getExploredArea(self):
		return self.drone.exploredSet


	##################### SARS structure End#####################

	


game = Simulator2(10,10,20,20,20,False)
#game.start()
#game.simulation()

#### EXAMPLE
# to start real game and get SARS
iters = 10
for i in range(10):
	print ('\niter: ', i, '\n')

	state = game.getState()
	print ('state: \n', state)

	exploredSet = game.getExploredArea()
	print ('explored area: \n', exploredSet)

	gridValue = game.getGrid()
	print ('value of grid: \n', gridValue )

	possibleActions = game.getAction(state)
	print ('possible actions: \n', possibleActions)	

	for action in possibleActions:
		print ('action: ', action)
		nextState, reward = game.getNextStateAndReward(state,action)
		print ('	reward, nextState: \n', reward, nextState)

		probability = game.getPossibility(state,action,nextState)
		print ('	probability: ', probability)

	# action = random.choice(possibleActions)
	action = input(">>> next action: ")
	print ('take action: ', action)
	game.moveAction(action)




