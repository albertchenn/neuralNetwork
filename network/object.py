# object.py

# number of each node
from network import myFirstNetwork


inputNodes = 3
hiddenNodes = 3
outputNodes = 3

# setting learning rate
learningRate = 0.3

myNetwork = myFirstNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
print(myNetwork)