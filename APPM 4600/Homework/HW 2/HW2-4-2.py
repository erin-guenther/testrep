import numpy as np
import math
import matplotlib.pyplot as plt

# for 0 ≤ θ ≤ 2π and for R = 1.2, δr = 0.1, f = 15 and p = 0. Make sure to adjust the
# scale so that the axis have the same scale.
# In a second figure, use a for loop to plot 10 curves and let with R = i, δr = 0.05,
# f = 2 + i for the ith curve. Let the value of p be a uniformly distributed random number
# (look up random.uniform) between 0 and 2.

theta = np.linspace(0,2*math.pi,100)
R = 1.2
delta_r = 0.1
f = 15
p = 0

x = R*(1 + delta_r*np.sin(f*theta+ p)) *np.cos(theta)
y = R*(1 + delta_r*np.sin(f*theta+ p)) *np.sin(theta)

plt.plot(x,y)
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()
