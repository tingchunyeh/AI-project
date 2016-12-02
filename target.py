from random import randint
import numpy as np
import random
class Target:

	name = 'target'
	rewardRange = 2
	maxReward = 10

	def __init__(self,x,y,randMove=True):
		self.x = x
		self.y = y
		self.lastX = x
		self.lastY = y
		self.backX = x
		self.backY = y
		self.backLx = x
		self.backLy = y
		self.faceAngle = 0 # 3 possible choice (+0(face front), +90(face left), -90 degrees(face right))
		self.lastFaceAngle = 0
		self.face = 180 # 4 possible condition( 0(face east),90(north), 180(west), 270(south))]
		self.lastFace = 180
		self.backF = 180
		self.backLF = 180
		self.backFA =  0
		self.backLFA = 0
		self.randMove = randMove
		##	Reward rule:
		#				 0.7 0.8 	0.45  0.1 0.05
		#				 0.9 1.0 	0.6   0.2 0.1
		#				 0.9 1.0 <-target 0.2 0.1
		#			     0.9 1.0    0.6   0.2 0.1
		#				 0.7 0.8	0.45  0.1 0.05
		# make reward matrix basd on assumption 
		# which means target face west and face front 
		self.rewardMatrix = [180,np.array([	[0.7, 0.8, 0.45, 0.2, 0.05],
											[0.9, 1.0, 0.6,  0.2, 0.1],
											[0.9, 1.0, -2, 0.2, 0.1],
											[0.9, 1.0, 0.6,  0.2, 0.1],
											[0.7, 0.8, 0.45, 0.1, 0.05],
										],float)] # first 180 indicate degree

		self.rewardMatrix[1] = np.multiply(self.rewardMatrix[1],self.maxReward)
	

	# targer doesn'y move but instead look left or right
	def moveFace(self):
		self.lastFaceAngle = self.faceAngle
		# if target currently look ahead
		if self.faceAngle == 0:
			if randint(0,1)==0:
				self.faceAngle = -90 #(face right)
			else: 
				self.faceAngle = 90  # face left
		# if target currently left or right
		else:
			self.faceAngle = 0

	# target move, record its last step
	def walk(self,newX, newY):
		self.lastX,self.lastY, self.lastFace= self.x,self.y, self.face
		self.x, self.y = newX, newY
		if self.x - self.lastX > 0:
			self.face = 0 # east
		elif self.x - self.lastX < 0:
			self.face = 180 # west
		else:
			if self.y - self.lastY >0:
				self.face = 270 # south
			else:
				self.face = 90 # north

	def preferenceMove(self,grids_x,grids_y):
		# until find a valid move
		while True:
			# random chance that target change face direction
			if randint(1,10) == 0:
				self.moveFace()
				break
			# random chance to move
			# moveX,moveY = randint(-1,1),randint(-1,1)
			moveX,moveY = random.choice([(1,0),(0,1),(0,-1)])
			newX,newY = self.x+moveX,self.y+moveY
			# check boundary (never walk along the boundary, at least one block away from boundary)
			if (newX>=1 and newX<=grids_x-2) and (newY>=1 and newY<=grids_y-2):
				self.walk(newX, newY)
				break
		return (moveX,moveY)

	# target decide to move or look around at still
	def randmove(self,grids_x,grids_y):
		# until find a valid move
		while True:
			# random chance that target change face direction
			if randint(1,10) == 0:
				self.moveFace()
				break
			# random chance to move
			# moveX,moveY = randint(-1,1),randint(-1,1)
			moveX,moveY = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
			newX,newY = self.x+moveX,self.y+moveY
			# check boundary (never walk along the boundary, at least one block away from boundary)
			if (newX>=1 and newX<=grids_x-2) and (newY>=1 and newY<=grids_y-2):
				self.walk(newX, newY)
				break
		return (moveX,moveY)

	def move(self,grids_x,grids_y):
		if self.randMove:
			return self.randmove(grids_x,grids_y)
		else:
			return self.preferenceMove(grids_x,grids_y)

	# according to input location of targey (x,y), return reward
	def computeRewardMatrix(self):
		x, y = self.x, self.y
		face,faceAngle = self.face, self.faceAngle
		size = 2*self.rewardRange+1
		# rewardMatrix = [ [0 for x in range(size)] for  y in range(size)]
		faceDirection = face + faceAngle
		faceDirection = (faceDirection+360) % 360
		self.rewardMatrix[0] = self.rewardMatrix[0] % 360
		# rotate to correct direction
		while self.rewardMatrix[0]%360 != faceDirection:
			self.rewardMatrix[1] = np.rot90( self.rewardMatrix[1])
			self.rewardMatrix[0] += 90

	# return reward of a specific location
	def getReward(self,x,y):
		if not self.isReward(x,y):
			return 0
		relative_x = x-self.x + self.rewardRange
		relative_y = y-self.y + self.rewardRange 
		return self.rewardMatrix[1][relative_y][relative_x]
		
	# whether the input location is be rewarded
	def isReward(self,x,y):
		diff_x = self.x - x
		diff_y = self.y - y
		if abs(diff_x)>self.rewardRange or abs(diff_y)>self.rewardRange :
			return False
		return True

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
		self.backLx, self.backLy = self.lastX, self.lastY
		self.backF, self.backLF = self.face, self.lastFace
		self.backFA, self.backLFA = self.faceAngle, self.lastFaceAngle

	def recoverBackUp(self):
		self.x, self.y = self.backX, self.bakcY
		self.lastX, self.lastY = self.backLx, self.backLy
		self.face, self.lastFace = self.backF, self.backLF
		self.faceAngle, self.lastFaceAngle = self.backFA, self.backLFA

	def AImove(self,action):
		self.lastX,self.lastY= self.x,self.y
		self.lastFace, self.lastFaceAngle = self.face, self.faceAngle

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
		if self.x - self.lastX > 0:
			self.face = 0 # east
		elif self.x - self.lastX < 0:
			self.face = 180 # west
		else:
			if self.y - self.lastY >0:
				self.face = 270 # south
			else:
				self.face = 90 # north
				

	def undo(self):
		self.x = self.lastX
		self.y = self.lastY
		self.faceAngle = self.lastFaceAngle
		self.face = self.lastFace
		self.computeRewardMatrix()

# agent = Target(1,3)
# print agent.rewardMatrix
# print (agent.x, agent.y, agent.face, agent.faceAngle)
# print 'move-------------'
# agent.randmove(20,20)
# print (agent.lastX, agent.lastY, agent.lastFace, agent.lastFaceAngle)
# print (agent.x, agent.y, agent.face, agent.faceAngle)
# agent.computeRewardMatrix(agent.x, agent.y, agent.face, agent.faceAngle)
# print agent.rewardMatrix
# print 'move-------------'
# agent.randmove(20,20)
# print (agent.lastX, agent.lastY, agent.lastFace, agent.lastFaceAngle)
# print (agent.x, agent.y, agent.face, agent.faceAngle)
# agent.computeRewardMatrix(agent.x, agent.y, agent.face, agent.faceAngle)
# print agent.rewardMatrix




