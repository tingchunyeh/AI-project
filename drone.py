import pygame
class Drone:

	name = 'drone'
	
	visionRange = 2

	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.lastX = x
		self.lastY = y 
		self.exploredSet = set()
		self.lastExploration = set()
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
		if self.x >= GRIDS_X-1:
			return ['west','north','south']
		elif self.y >= GRIDS_Y-1:
			return ['west','east','north']
		elif self.x <= 0:
			return ['north','east','south']
		elif self.y <= 0:
			return ['west','east','south']
		else:
			return ['west','east','south','north']


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
		self.expandExploration()
		






