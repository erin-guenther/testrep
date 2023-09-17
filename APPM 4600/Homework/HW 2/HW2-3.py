import math
import numpy as np

def driver1(x):
    y = math.exp(x)
    return y-1

def driver2(x):
    y = x + x**2/2 
    return y

x = 9.999999995000000 * 10**-10
actual = 1*10**-9
y = driver2(x)

error = abs(y-actual)/abs(actual)
print(y)
print("the relative error is:",error)