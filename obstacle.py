class Obstacle:

	reward = -50
	name = 'obstacle'
	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.lastX = 0
		self.lastY = 0


