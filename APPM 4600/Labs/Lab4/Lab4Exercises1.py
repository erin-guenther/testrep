# 3.1: we have p = (-2 * p(n+1) +p(n) + p(n+1))/(-(p(n+1))**2 + p(n) + p(n+2))

#3.2
import numpy as np
import math

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    x = np.zeros((Nmax,1))

    count = 0
    while (count <Nmax):
       x[count] = x0
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0


        #clear 0's at end
          i = Nmax-1
          while (i >= count) and (i<= Nmax-1):
            x = np.delete(x,i)
            i = i - 1

          return [xstar,ier,x,count]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier,x,count]

def better_approx(x, ier, tol):
   # pass vector in from fixed point iteration and apply new approximation

    length = len(x)-2 #need to subtract 2 so we don't have an indexing issue later
    y = np.zeros((length,1))

    i = 1
    while i<length:

        if i>1 and abs(y[i] - y[i-1])<tol :
            return [y,i]
        
        y[i] = (-(x[i+1]**2) + x[i] + x[i+2])/(-2*x[i+1]+x[i]+x[i+2])
        #debugging
        print(y[i])

        i += 1
    
    return [y,i]



def driver():
    # test functions 
     f1 = lambda x: (10/(x+4))**(0.5)

     Nmax = 20
     tol = 1e-10
     x0 = 1.5

     [xstar,ier,x,count] = fixedpt(f1,x0,tol,Nmax)
     #print('the approximate fixed point is:',xstar)
     #print('f4(xstar):',f1(xstar))
     print(x)
     #print("The number of iterations it took was:", count)

     [y,i]= better_approx(x,ier,tol)
     print(y)
     print("The number of iterations it took was:", i)


driver()

#Aitken's method does converge faster: Fixed point took 12 iterations while Aitken's only took 10.