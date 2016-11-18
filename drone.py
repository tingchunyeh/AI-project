import pygame
class Drone:

	name = 'drone'

	visionRange = 2

	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.lastX = x
		self.lastY = y
		self.backX = x
		self.backY = y
		self.backLx = x
		self.backLy = y
		self.exploredSet = set()
		self.lastExploration = set()
		self.backExp = set()
		self.backLexp = set()
		self.expandExploration()

	# according to keyboard input, move position
	def controlMove(self,event):
		self.lastX = self.x
		self.lastY = self.y
		if event.key == pygame.K_LEFT:
			self.x -= 1
		if event.key == pygame.K_RIGHT:
			self.x += 1
		if event.key == pygame.K_UP:
			self.y -= 1
		if event.key == pygame.K_DOWN:
			self.y += 1
		self.expandExploration()


	def expandExploration(self):
		self.lastExploration = set(self.exploredSet)
		for x in range(self.x-self.visionRange,self.x+self.visionRange+1):
			for y in range(self.y-self.visionRange,self.y+self.visionRange+1):
				self.exploredSet.add((x,y))

	def checkExploration(self,x,y):
		return (x,y) in self.exploredSet

	def undo(self):
		self.x = self.lastX
		self.y = self.lastY
		self.exploredSet = self.lastExploration

	def possibleActions(self,GRIDS_X,GRIDS_Y):
		actions = []

		if self.x < GRIDS_X-1:
			actions.append('east')
		if self.x > 0:
			actions.append('west')
		if self.y < GRIDS_Y-1:
			actions.append('south')
		if self.y > 0:
			actions.append('north')

		return actions

	def backUp(self):
		self.backX, self.bakcY = self.x, self.y
		self.backExp = set(self.exploredSet)
		self.backLx, self.backLy = self.lastX, self.lastY
		self.backLexp = set(self.lastExploration)

	def recoverBackUp(self):
		self.x, self.y = self.backX, self.bakcY
		self.exploredSet = set(self.backExp)
		self.lastX, self.lastY = self.backLx, self.backLy
		self.lastExploration = set(self.backLexp)

	def AImove(self,action):
		self.lastX = self.x
		self.lastY = self.y
		if action == 'east':
			self.x += 1
		elif action == 'west':
			self.x -= 1
		elif action == 'north':
			self.y -= 1
		elif action == 'south':
			self.y += 1
		else:
			None
		#print (self.x,self.y)
		self.expandExploration()
