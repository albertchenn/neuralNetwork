import matplotlib.pyplot as plt
import numpy as np

array = np.array([[1, 3], [2, 6]])

plt.imshow(array, interpolation='nearest')
plt.savefig("learning/outputs/1326.png")

