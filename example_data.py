import numpy as np
import os

# Small script to generate linear data with normally distributed noise for one independent variable
# n specifies n dependent variables and an additional column denoting a single independent variable is added
# This column lists the positive integers from 1 to r
# r specifies the number of data points
# Two random constants between -10 and 10 are generated, representing the two constants defining the line
# Random noise is added to this

n = 5
r = 50

noise = np.random.normal(0, 30, size=(r, n))
independent = np.arange(1, r + 1)
data = np.zeros((r, n + 1))
data[:, 0] = independent.transpose()
data[:, 1:] = noise

for i in range(1, n + 1):
    uniform_1 = np.random.uniform(low=-10, high=10, size=(1,))
    uniform_2 = np.random.uniform(low=-10, high=10, size=(1,))
    for j in range(r):
        data[j, i] += uniform_1 + uniform_2*data[j,0]

path = os.getcwd()
np.savetxt(path + r"\data.dat", data)




