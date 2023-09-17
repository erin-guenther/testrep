import numpy as np
import math
import matplotlib.pyplot as plt


for i in range(10):
    theta = np.linspace(0,2*math.pi,100)
    R = 1.2
    delta_r = 0.1
    f = 15
    p = 0

    R = i
    delta_r = 0.05,
    f = 2 + i 
    p = np.random.uniform(0,2)

    x = R*(1 + delta_r*np.sin(f*theta+ p)) *np.cos(theta)
    y = R*(1 + delta_r*np.sin(f*theta+ p)) *np.sin(theta)
    plt.plot(x,y)

    
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.show()