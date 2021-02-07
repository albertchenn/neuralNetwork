# object.py

# number of each node
from network import myFirstNetwork

# number of each node
inputNodes = 3
hiddenNodes = 3
outputNodes = 3

# setting learning rate
learningRate = 0.3

# create network
myNetwork = myFirstNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# query it with random numbers to test query()

output = myNetwork.query([1.0, 0.5, -1.5])
print(output)