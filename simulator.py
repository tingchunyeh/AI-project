from drone import Drone
from target import Target
from obstacle import Obstacle
import pygame
from random import randint
import numpy as np
import time


class Simulator:

	# Define some colors
	BLACK = (0, 0, 0)
	WHITE = (255, 255, 255)
	RED = (255, 0, 0)
	BLUE = (0,0,255)
	GREEN = (0,225,0)
	YELLOW = (225,225,0)

	# Initialize the setting of world
	def __init__(self,grids_x=15,grids_y=15,grid_width=20,grid_height=20,obstaclesNum=25):
		self.GRIDS_X = grids_x # how many squares in x direction
		self.GRIDS_Y = grids_y # how many squares in y direction
		self.WIDTH = grid_width # square width
		self.HEIGHT = grid_height # square height
		self.OBSTACLSNUM = obstaclesNum # how many obstacles in the setting
		self.MARGIN = 1 # how long between each two squares
		self.grid = [ [0 for x in range(grids_x)] for y in range(grids_y) ]
		self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		self.obstaclesLs = []
		self.offsetY = 40
		self.score = 0


	# randomly generate strat point on the grid for every stuff
	def generateStartPoint(self):
		locationSet = set()
		while len(locationSet) < self.OBSTACLSNUM+1: # +1 for drone
			locationSet.add( (randint(0,self.GRIDS_X-1),randint(0,self.GRIDS_Y-1)) )

		count = 0
		for loca in locationSet:
			loca = list(loca)
			count += 1
			x,y = loca[0],loca[1]
			# for obstacles
			if count <= self.OBSTACLSNUM:
				block = Obstacle(x,y)
				self.obstaclesLs.append(block)
				self.grid[y][x] = block
			# for drone
			else:
				drone = Drone(x,y)
				self.drone = drone
				self.grid[y][x] = drone
		# for target
		x,y = randint(2,self.GRIDS_X-3),randint(2,self.GRIDS_Y-3)
		target = Target(x,y)
		self.target = target
		self.grid[y][x] = target

 	#after every stuff move, update the information of grid and value of grid
	def updateGrid(self):
		# remove original location
		target = self.target
		self.grid[target.lastY][target.lastX] = 0
		drone = self.drone
		self.grid[drone.lastY][drone.lastX] = 0

		# for obstacles
		for obstacle in self.obstaclesLs:
			lastX,lastY = obstacle.lastX,obstacle.lastY
			x,y = obstacle.x,obstacle.y
			self.grid[lastY][lastX] = 0
			self.grid[y][x] = obstacle

		# for Target
		self.grid[target.y][target.x] = target
		# for drone
		self.grid[drone.y][drone.x] = drone

	def computeValue(self,firstStep):
		# self.gridValue = np.asarray( [ [0 for x in range(self.GRIDS_X)] for y in range(self.GRIDS_Y) ] )
		# calculate obstacle value
		for obstacle in self.obstaclesLs:
			self.gridValue[obstacle.y][obstacle.x] = obstacle.reward

		# target's reward
		target = self.target
		# remove last step
		if not firstStep:
			self.mapRewardMatrixtoGrid(target.lastX,target.lastY,target,'minus')
		self.mapRewardMatrixtoGrid(target.x,target.y,target,'add')


	def mapRewardMatrixtoGrid(self,x,y,target,do):
		rewardRange = target.rewardRange
		startX = x - rewardRange
		endX = x + rewardRange
		startY = y - rewardRange
		endY = y + rewardRange
		if do == 'add':
			target.computeRewardMatrix(target.x,target.y,target.face,target.faceAngle)
		else:
			target.computeRewardMatrix(target.lastX,target.lastY,target.lastFace,target.lastFaceAngle)

		rewardMatrix = target.rewardMatrix[1]
		for row_grid,row_reward in zip(range(startY,endY+1),range(2*rewardRange+1)):
			if row_grid < 0 or row_grid >=self.GRIDS_Y:	continue
			for col_grid, col_reward in zip(range(startX,endX+1),range(2*rewardRange+1)):
				if col_grid < 0 or col_grid >=self.GRIDS_X:	continue
				if do == 'add':
					self.gridValue[row_grid][col_grid] += rewardMatrix[row_reward][col_reward]
				else:
					self.gridValue[row_grid][col_grid] -= rewardMatrix[row_reward][col_reward]


	def drawSquares(self,screen):
		yellow = np.asarray(list(self.YELLOW))
		white = np.asarray(list(self.WHITE))
		gradient = np.subtract(yellow,white)
		for row in range(self.GRIDS_Y):
 			for col in range(self.GRIDS_X):
 				DistLeft = (self.MARGIN + self.WIDTH) * col + self.MARGIN
 				DistUp = (self.MARGIN + self.HEIGHT) * row + self.MARGIN
 				if self.grid[row][col] == 0:
 					color = self.WHITE
 					if self.target.isReward(col,row):
 						w = self.target.getReward(col,row)/self.target.maxReward
 						colorls = np.add(white,np.multiply(gradient,w))
 						color = tuple(colorls)
 				else:
 					obj = self.grid[row][col]
 					name = obj.name
 					# if target.reward(col,row)!=0:
 					if name is 'obstacle':
 						color = self.RED
 					if name is 'target':
 						color = self.GREEN
 					if name is 'drone':
 						color = self.BLUE

 				pygame.draw.rect(screen,color,[DistLeft,DistUp+self.offsetY,self.WIDTH,self.HEIGHT])


	def draw_text(self, text, font, surface, x, y, main_color, background_color=None):
		textobj = font.render(text, True, main_color, background_color)
		textrect = textobj.get_rect()
		textrect.centerx = x
		textrect.centery = y
		surface.blit(textobj, textrect)



	def eventHandler(self,screen,firstStep):
		for event in pygame.event.get():
			# If user clicked close
			if event.type == pygame.QUIT:
				done = True
			if event.type == pygame.KEYDOWN:
				self.drone.controlMove(event,self.GRIDS_X,self.GRIDS_Y)
			self.updateGrid()
			self.score += self.gridValue[self.drone.y][self.drone.x]

	def targetMoveTiming(self,elapsed,lastMoveTime,timePeriod,firstStep,screen):
		if (elapsed - lastMoveTime) > timePeriod or firstStep:
			print ('move')
			lastMoveTime = elapsed
			self.target.randmove(self.GRIDS_X,self.GRIDS_Y)
			self.computeValue(firstStep)
			firstStep = False
			self.updateGrid()
			self.score += self.gridValue[self.drone.y][self.drone.x]
		return lastMoveTime,firstStep

	def canvas(self):
		windowSize = [ (self.GRIDS_X+self.MARGIN)*self.WIDTH, (self.GRIDS_Y+self.MARGIN)*self.HEIGHT+self.offsetY]
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

	def moveSametime(event,done):
		for event in pygame.event.get():
			# If user clicked close
			if event.type == pygame.QUIT:
				return done
			if event.type == pygame.KEYDOWN:
				self.drone.controlMove(event,self.GRIDS_X,self.GRIDS_Y)
				self.targetMoveTiming(elapsed,lastMoveTime,timePeriod,firstStep,screen)
			self.updateGrid()
			self.score += self.gridValue[self.drone.y][self.drone.x]


	def start(self):
		print ("game")
		pygame.init()
		screen = self.canvas()
		default_font = pygame.font.Font(None, 28)

		# Used to manage how fast the screen updates
		timePeriod = 0.5
		timeEnd = 30

		start_time = time.time()
		lastMoveTime = start_time
		clock = pygame.time.Clock()
		
		# Loop until the user clicks the close button.
		firstStep = True
		done = False
		while not done:
			screen.fill(self.BLACK)
			elapsed = time.time() - start_time
			# self.eventHandler(screen,firstStep)
			# lastMoveTime,firstStep = self.targetMoveTiming(elapsed,lastMoveTime,timePeriod,firstStep,screen)
			done = moveSametime(event,done)
			# draw
			self.drawScore(screen,default_font,elapsed,timeEnd)

		    # Limit to 60 frames per second
			clock.tick(60)
		    # Go ahead and update the screen with what we've drawn.
			pygame.display.flip()
			if elapsed > timeEnd:
				print (self.score)
				break
			print (self.gridValue,'\n')
		# pygame.quit()


game = Simulator()
game.start()
