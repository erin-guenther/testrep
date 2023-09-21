# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**(0.5)

     Nmax = 100000
     tol = 1e-10

     x0 = 1.5
     [xstar,ier,x,count] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f4(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print(x)
     print("The number of iterations it took was:", count)



# define routines
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

    

driver()

#Prelab: It took 12 iterations to converge, and we know the fixed point method has a quadratic convergence