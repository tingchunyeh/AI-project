from game import Game
import pygame

game = Game(10,10,20,20,20,False)
# game.start()
# game.simulation()

# to start real game and get SARS
############### SARS EXAMPLE ###################
iters = 10
for i in range(10):
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
	
		
	# action = random.choice(possibleActions)
	action = input(">>> next action: ")
	print ('take action: ', action)
	game.moveAction(action)

