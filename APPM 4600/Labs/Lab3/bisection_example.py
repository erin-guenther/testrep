# import libraries
import numpy as np

def driver():

# Q1:
#(c) f (x) = sin(x) with a = 0, b = 0.1. What about a = 0.5 and b = 3π
  f = lambda x: np.sin(x)
  (a,b) = (0.5,3*np.pi)

  tol = 1e-5

  [astar,ier] = bisection(f,a,b,tol)
  print('the approximate root is',astar)
  print('the error message reads:',ier)
  print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               

# Solutions:
#Q1(a): 
# 
# the approximate root is 0.9999999701976776
# the error message reads: 0
# f(astar) = -2.98023206113385e-08

#Q1(b):
# 
# the approximate root is -1
# the error message reads: 1
# f(astar) = -2

#Q1(c):
# the approximate root is 0.9999999701976776
# the error message reads: 0
# f(astar) = -2.98023206113385e-08

#Q1 discussion: It isn't possible to find the root x=0 using bisection due to the fact that the root has a multiplicity of 2.
# (This means that the function does not cross the axis). The intervals in (a) and (c) work because they contain a root, but (b)
# does not because there is no root with odd multiplicity within the interval (-1,0.5).

#Q2(a):
# the approximate root is 1.0000030517578122
# the error message reads: 0
# f(astar) = 2.4414006618542327e-05

#Q2(b):
# the approximate root is 0
# the error message reads: 1
# f(astar) = -3

#Q2(c):
# the approximate root is 0
# the error message reads: 0
# f(astar) = 0.0

#
