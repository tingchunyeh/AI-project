from game import Game
import pygame
import time




def timesLimitedStart(timeEnd):
	game = Game(10,10,20,20,30,'./environment/obstaclesMap2.txt')
	start_time = time.time()
	lastMoveTime = start_time
	# Loop until the user clicks the close button.
	done = False
	while not done:
		game.screen.fill(game.BLACK)
		elapsed = time.time() - start_time
		done = game.eventHandler()
		game.drawScore(game.screen,game.default_font,elapsed,timeEnd) # draw
		pygame.display.flip() # Go ahead and update the screen with what we've drawn.
		if elapsed > timeEnd:
			print (game.score)
			break
	pygame.quit()

# timesLimitedStart(30)

def numbersLimitedStart(iters):
	game = Game(10,10,20,20,30,'./environment/obstaclesMap2.txt')
	for i in range(1,iters+1):
		game.drawCanvas(i,iters)
		action = True
		while action:
			action = game.eventWait()

	print (game.score)
	pygame.quit()

# numbersLimitedStart(50)




	############### SARS EXAMPLE ###################
def SARSExample():
	# to start real game and get SARS
	game = Game(10,10,20,20,30,'./environment/obstaclesMap2.txt')
	iters = 10
	for i in range(iters):
		print ('\niter: ', i, '\n')
		game.drawCanvas(i,iters)

		state = game.getState()
		print ('state: \n', state)

		exploredSet = game.getExploredArea()
		print ('explored area: \n', exploredSet)

		gridValue = game.getGrid()
		print ('value of grid: \n', gridValue )

		possibleActions = game.getAction(state)
		print ('possible actions: \n', possibleActions) 

		for action in possibleActions:
			print ('	action:',action)
			successors = game.getSuccessors(state,action)
			print ('successors are:\n', successors)
			# nextStates,rewards = game.getNextStateAndReward(state,action)
			# for nextState,reward in zip(nextStates,rewards):
			# 	probability = game.getPossibility(state,action,nextState)
			# 	print ('		nextState:',nextState,'; reward:',reward,'; probability: ', probability)
		print ('value of grid:', game.getRewardMatrix(1))
		action = input(">>> next action: ")
		print ('take action: ', action)
		game.moveAction(action)
	print (game.score)
	pygame.quit()

SARSExample()
# for i in range(10):
# 	numbersLimitedStart(50)
# 	pygame.quit()




