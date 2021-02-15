# network.py

# imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

class myFirstNetwork():

    # initalize the network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate) -> None:
        # set number of nodes in each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        # learning rate
        self.learningRate = learningRate

        # set initial weights (1 / sqrt(number of nodes in next layer))
        self.inputHiddenWeightsMatrix = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.hiddenOutputWeightsMatrix = np.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        # set activation function

        self.activationFunction = lambda x: sp.expit(x)
        pass

    # training function
    def train(self, inputsList, targetsList):
        # make the inputs and targets list a 2D array using numpy
        inputs = np.array(inputsList, ndmin=2).T
        targets = np.array(targetsList, ndmin=2).T

        # find signals being inputted into hidden layer
        hiddenInputs = np.dot(self.inputHiddenWeightsMatrix, inputs)
        # find signals being outputted from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # find signals being inputted into final layer
        finalInputs = np.dot(self.hiddenOutputWeightsMatrix, hiddenOutputs)
        # find signals being outputted from final layer
        finalOutputs = self.activationFunction(finalInputs)
        
        # find the errors (difference from targets)
        errors = targets - finalOutputs

        # split the errors and recombine at each hidden node
        hiddenErrors = np.dot(self.hiddenOutputWeightsMatrix.T, errors) 

        # update the weights from the hidden to output layers
        self.hiddenOutputWeightsMatrix += self.learningRate * np.dot((errors * finalOutputs * (1.0 - finalOutputs)), np.transpose(hiddenOutputs))
        
        # update the weights from the input to hidden layers
        self.inputHiddenWeightsMatrix += self.learningRate * np.dot((errors * hiddenOutputs * (1.0 - hiddenOutputs)), np.transpose(inputs))
        
    # query function
    def query(self, inputsList):
        # make the inputs list a 2D array using numpy
        inputs = np.array(inputsList, ndmin=2).T

        # find signals being inputted into hidden layer
        hiddenInputs = np.dot(self.inputHiddenWeightsMatrix, inputs)
        # find signals being outputted from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # find signals being inputted into final layer
        finalInputs = np.dot(self.hiddenOutputWeightsMatrix, hiddenOutputs)
        # find signals being outputted from final layer
        finalOutputs = self.activationFunction(finalInputs)
        
        return finalOutputs


