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
import sklearn
import sklearn.datasets
import sklearn.linear_model


# Model Based Monte Carlo
# requires data to estimate transitions and rewards

# Model Free Monte Carlo
# requires data to find optimal Q

# policy: uniform random
# At each state, need to go through each possible random action
# Goal: Estimate Q_opt at state, and action

# Needs a dictionary of (s, a): (rewards, num of updates starting at 0)



class neural:
    gamma = 0.7

    def __init__(self,game):
        self.game = game
        self.policy = {}
        self.x = game.GRIDS_X
        self.y = game.GRIDS_Y
        self.numIters = 10
        self.q_dict = {}
        self.neuralData_input = []
        self.neuralData_output = []

    def getData(self, state, numOfMoves):
        if numOfMoves > 0:
            stateKey = tuple(state)
            #print ("state:", state)
            possActions = self.game.getAction(state)

            action = random.choice(possActions)
            #print ("action", action)

            reward = self.game.moveAction(action)
            dictKey = (stateKey, action)
            if dictKey in self.q_dict:
                currReward, currNumUpdates = self.q_dict.get(dictKey)
                eta = 1.0/(1+currNumUpdates)
                newReward = (1-eta)*currReward + eta * reward
                self.q_dict[dictKey] = ( newReward , currNumUpdates+1 )
            else:
                self.q_dict[dictKey] = (reward, 1)
            #print ("new state:", self.game.getState())
            self.getData(self.game.getState(), numOfMoves-1)


    def getBestAction(self, state):
        possActions = self.game.getAction(state)

        stateKey = tuple(state)
        maxVal = float('-inf')
        bestAction = None
        for action in possActions:
            dictKey = (stateKey, action)
            if dictKey in self.q_dict:
                reward, numUpdates = self.q_dict.get(dictKey)
                if reward > maxVal:
                    maxVal = reward
                    bestAction = action
        if bestAction == None:
            return random.choice(possActions)
        else:
            return bestAction

nn_input_dim = 5 # input layer dimensionality
nn_output_dim = 4 # output layer dimensionality
epsilon = 0.00001 # learning rate for gradient descent
reg_lambda = 0.001 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    z2 = z2 - np.amax(z2) + 0.00000001
    #print (np.amax(z2))

    exp_scores = np.exp(z2)
    #print(exp_scores)
    #print(np.sum(exp_scores, axis=1, keepdims=True))
    #print (np.isfinite(exp_scores).all())
    exp_scores[exp_scores == 0] = 0.00000001    #print (z2)
    #print (exp_scores)
    #while 1:
    #    a=1
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    #print(probs[range(num_examples), y])
    #while 1:
    #    a=1
    #corect_logprobs = -np.log(probs[range(num_examples), y])
    corect_logprobs = -np.log(probs[np.where(y==1)])
    #corect_logprobs.filled(0)
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    print("W1 shape", W1.shape)
    print("b1 shape", b1.shape)
    print("W2 shape", W2.shape)
    print("b2 shape", b2.shape)

    print ("x", x)
    print("x shape", x.shape)
    z1 = x.dot(W1) + b1
    print("z1", z1)
    a1 = np.tanh(z1)
    print("a1", a1)
    z2 = a1.dot(W2) + b2
    print("z2 before",z2)
    z2 = z2 - np.amax(z2) + 0.00000001
    print("z2 after",z2)
    exp_scores = np.exp(z2)
    print ("exp_scores before", exp_scores)
    exp_scores[exp_scores == 0] = 0.00000001
    print ("exp_scores after", exp_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print ("probs", probs)
    #return np.argmax(probs, axis=1)
    print ("sorted",probs.argsort()[-4:][::-1])
    #while 1:
    #    a=1
    return probs.argsort()[-4:][::-1]

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, X, y, num_passes=2000, print_loss=False):

    num_examples = X.shape[0] # training set size

    # Initialize the parameters to random values. We need to learn these.
    #np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        # print("X shape", X.shape)
        # print("W1 shape", W1.shape)
        # print("b1 shape", b1.shape)
        # print("W2 shape", W2.shape)
        # print("b2 shape", b2.shape)
        # print("z1 shape", z1.shape)
        # print("a1 shape", a1.shape)
        # print("z2 shape", z2.shape)
        #print("z2 before",z2)

        z2 = z2 - np.amax(z2) + 0.00000001
        #print("z2 after",z2)

        #print (np.amax(z2))
        exp_scores = np.exp(z2)
        #print(exp_scores)
        #print(np.sum(exp_scores, axis=1, keepdims=True))
        #print (np.isfinite(exp_scores).all())
        exp_scores[exp_scores == 0] = 0.00000001
        #print (exp_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
        # Backpropagation
        delta3 = probs
        #print (delta3[range(num_examples), 3])

        #print ("y ", y)
        #print ("y shape", y.shape)
        #print("delta3 ", delta3)
        #print ("delta3 shape", delta3.shape)
        #print ("special", np.where(y==1) )
        #print(delta3[np.where(y==1) ])

        delta3[np.where(y==1)] -= 1

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        #print("z1",z1)
        #print("dd3",delta3.dot(W2.T))
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        #delta2 = delta3.dot(W2.T) * (1 - z1) * (1 + z1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1


        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
          print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def isObsMove(action, obs):
    # print(game.grid[myY][myX+1])
    # print(game.grid[myY-1][myX])
    # print(game.grid[myY][myX-1])
    # print(game.grid[myY+1][myX])
    if action == "east" and (1,0) in obs:
        return True
    elif action == "south" and (0,1) in obs:
        return True
    elif action == "west" and (-1,0) in obs:
        return True
    elif action == "north" and (0,-1) in obs:
        return True
    else:
        return False

######################################################################################################
training = True
game = Game(10,10,20,20,30,'obstaclesMap2.txt')
nn = neural(game)
trainDict = {}
######################################################################################################
if training:
    print('Training...')
    trainDict = np.load('actionDict.npy').item()

    # gather data
    for item in trainDict:
        bestAction = trainDict[item]
        #print(list(state))
        dataIn = [item[0][0], item[0][1], item[1][0], item[1][1], item[2][0]]
        #print (data)
        nn.neuralData_input.append(dataIn)
        #print (max(dataIn))

        dataOut = (0,0,0,0)
        if bestAction == "east":
            dataOut = (1,0,0,0)
        elif bestAction =="south":
            dataOut = (0,1,0,0)
        elif bestAction =="west":
            dataOut = (0,0,1,0)
        elif bestAction =="north":
            dataOut = (0,0,0,1)
        #print list(dataOut)
        #print (state)

        nn.neuralData_output.append(list(dataOut))

    X = np.asarray(nn.neuralData_input)
    #print (X)
    print("x shape", X.shape)
    # output dataset
    y = np.array(nn.neuralData_output)
    print("y shape", y.shape)
    num_examples = X.shape[0] # training set size
    #print (num_examples)

    # Build a model with a 3-dimensional hidden layer
    model = build_model(25, X, y, print_loss=True)
    print ("Model:",model)
    test = np.array( [  [9., 4., 1., 5., 90.],
                        [6., 0., 8., 2., 0.],
                        [2., 8., 1., 8., 270.],
                        [3., 1., 8., 4., 270.],
                        [9., 7., 2., 1., 180.],
                        [0., 2., 5., 5., 270.],
                        [2., 6., 3., 8., 180.]
                ] )

    print("Prediction:", predict(model, test) )

    # [[ 0.9   0.4   0.1   0.5   0.25]
    #  [ 0.6   0.    0.8   0.2   0.  ]
    #  [ 0.2   0.8   0.1   0.8   0.75]
    #  ...,
    #  [ 0.3   0.1   0.8   0.4   0.75]
    #  [ 0.9   0.7   0.2   0.1   0.5 ]
    #  [ 0.    0.2   0.5   0.5   0.75]]
    # [[0 0 0 1]
    #  [1 0 0 0]
    #  [0 0 0 1]
    #  ...,
    #  [1 0 0 0]
    #  [0 0 1 0]
    #  [0 1 0 0]]

    np.save('neuralModel_qLearn.npy', model)

    # Plot the decision boundary
    #plot_decision_boundary(lambda x: predict(model, x))
    #plt.title("Decision Boundary for hidden layer size 3")

###############################

    # #seed random numbers to make calculation
    # #deterministic (just a good practice)
    # np.random.seed(1)
    #
    # # initialize weights randomly with mean 0
    # syn0 = 2*np.random.random((6,4)) - 1
    #
    # for iter in range(10000):
    #
    #     # forward propagation
    #     l0 = X
    #     l1 = nonlin(np.dot(l0,syn0))
    #
    #     # how much did we miss?
    #     #print (l1.shape)
    #     #print (y.shape)
    #     l1_error = y - l1
    #
    #     # multiply how much we missed by the
    #     # slope of the sigmoid at the values in l1
    #     l1_delta = l1_error * nonlin(l1,True)
    #
    #     # update weights
    #     syn0 += np.dot(l0.T,l1_delta)
    #     if iter % 100==0:
    #         print ("Iter", iter)
    # print ("Output After Training:")
    # print (l1)

######################################################################################################
else:
    print("Testing...")
    model = np.load('neuralModel_qLearn.npy').item()

    numOfGames = 100
    gameStepLimit = 100
    totalScore = 0
    start = time.time()

    for j in range(numOfGames):
        game = Game(10,10,20,20,30,'obstaclesMap2.txt')
        nn.game = game
        for i in range(1,gameStepLimit+1):

            #print ('\niter: ', i)
            #game.drawCanvas(i,gameStepLimit)

            #print(model)

            state = game.getState()
            #print ('state: \n', state)

            #print (state)
            inputData = np.array( [ [state[0][0], state[0][1], state[1][0], state[1][1], state[2][0]] ] )
            #print(inputData)
            predictActions = predict(model, inputData)
            #print(bestAction)
            possActions = game.getAction(state)
            #print ('possible actions: \n', possActions)

            myX = state[0][0]
            myY = state[0][1]

            obsTemp = game.getObstaclesAronud()
            obs=[]
            for item in obsTemp:
                if item[1] == 1:
                    obs.append(item[0])

            actionList = []
            #print (predictActions)
            for act in predictActions[0]:
                if act == 0 and "east" in possActions:
                    if (1,0) not in obs:
                        actionList.append("east")
                elif act == 1 and "south" in possActions:
                    if (0,1) not in obs:
                        actionList.append("south")
                elif act == 2 and "west" in possActions:
                    if (-1,0) not in obs:
                        actionList.append("west")
                elif act == 3 and "north" in possActions:
                    if (0,-1) not in obs:
                        actionList.append("north")

            #print (obs)

            if len(actionList) > 0:
                action = actionList[0]
            else:
                action = random.choice(possActions)
            while isObsMove(action, obs):
                #print("yes")
                action = random.choice(possActions)
            #time.sleep(1)

            # AI action
            #action = nn.getBestAction(state)
            #print ('take action: ',action)
            game.moveAction(action)
        totalScore += game.getScore()
    end = time.time()
    print ("Testing time:", (end - start) / numOfGames)
    averageGameScore = float(totalScore/numOfGames)
    print ("Average game score", averageGameScore)
