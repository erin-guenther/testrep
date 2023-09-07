
import numpy as np
import math
import matplotlib.pyplot as plt

# (1)
# initialize 50 equally spaced values between 0 and pi using two different built in functions
x = np.linspace(0, np.pi, 50)
y = np.arange(0,np.pi, 50)

#(2)/(3)
# list the first 3 entries of the x array using splicing
print("The first 3 entires of x are", x[0:3])

# (4)
# create an array of 10 to the power of -1 to -10
w = 10**(-np.linspace(1,10,10))

# create a new array of integers from 1 to the length of the w array
x = np.arange(1,len(w)+1)

#plot x on the x-axis and w on the y-axis and label the axes
plt.plot(x,w)
plt.ylabel('w')
plt.xlabel('x')

# (5)
s =  3 * w
plt.plot(x,s)
plt.show()

