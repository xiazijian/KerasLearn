import numpy as np
x = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
y = np.array([[1,-1,-1],
              [-1,1,-1],
              [-1,-1,1]])
print(x.dot(y))