
import numpy as np
import math
import matplotlib.pyplot as plt

# (1)
x = np.linspace(0, np.pi, 50)
y = np.arange(0,np.pi, 50)

#(2)/(3)
print("The first 3 entires of x are", x[0:3])

# (4)
w = 10**(-np.linspace(1,10,10))

x = np.arange(1,len(w)+1)

plt.plot(x,w)
plt.ylabel('w')
plt.xlabel('x')

# (5)
s =  3 * w
plt.plot(x,s)
plt.show()

