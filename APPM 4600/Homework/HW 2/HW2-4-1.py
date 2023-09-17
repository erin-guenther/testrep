import math
import numpy as np
import matplotlib.pyplot as plt

# 0.001 is added to pi since this function is non-inclusive (this way pi is a value in the vector)
t = np.arange(0,math.pi+.001 ,math.pi/30) 
y = np.cos(t)

# initialize S before the loop
S = 0

# create a running sum of each element multiplied together
for i in range(len(t)):
    S += t[i]*y[i]

print("The sum is:", S)

