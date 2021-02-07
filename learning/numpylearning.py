import numpy as np

a = np.array([[3, 2],
              [1, 3],
              [6, 3]])

a[0,1] = 1

b = np.random.rand(3, 5) - 0.5

c = pow(2, -0.5)

d = np.array([1, 2, 3, 4, 5, 6], ndmin=2).T
# print(a)
# print(b)
# print(c)
print(d)
