import math
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.920, 2.080,0.001)

# x9 −18x8 +144x7 −672x6 +2016x5 −4032x4 +5376x3 −4608x2 +2304x−512

f = x**9 - 18*x**8 + 144*x**7 -672*x**6 + 2016*x**5 -4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
g = (x-2)**9

plt.plot(x,f)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Expanded Polynomial')
plt.show()


plt.plot(x,g)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Factored Polynomial')
plt.show()