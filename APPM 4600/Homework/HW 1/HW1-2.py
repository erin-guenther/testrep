import math
import numpy as np
import matplotlib.pyplot as plt

x = np.array([10**(-16), 10**(-15), 10**(-14), 10**(-13), 10**(-12),10**(-11),10**(-10),10**(-9),10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1])

#pi
y1 = 2*np.sin(-math.pi-x * 0.5)*np.sin(x/2)
y2 = np.cos(math.pi + x) - np.cos(math.pi)
ydiff1 = y1-y2

#10^6
y3 = 2*np.sin(-10**6- x * 0.5)*np.sin(x/2)
y4 = np.cos(10**6 + x) - np.cos(10**6)
ydiff2 = y3-y4

ydiff = ydiff1-ydiff2

plt.plot(np.log(x),y1)
plt.plot(np.log(x),y2)
plt.xlabel('x')
plt.xlim(-16,1)
plt.ylabel('y')
plt.title('Using x=pi')
plt.show()

plt.plot(np.log(x),y3)
plt.plot(np.log(x),y4)
plt.xlabel('x')
plt.xlim(-16,1)
plt.ylabel('y')
plt.title('Using x = 10**6')
plt.show()

plt.plot(np.log(x),ydiff1)
plt.xlabel('x')
plt.xlim(-16,1)
plt.ylabel('y')
plt.title('Plot of Differences for x=pi')
plt.show()

plt.plot(np.log(x),ydiff2)
plt.xlabel('x')
plt.xlim(-16,1)
plt.ylabel('y')
plt.title('Plot of Differences x=10**6')
plt.show()
