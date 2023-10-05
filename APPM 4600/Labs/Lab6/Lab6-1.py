
#3.2.1 - one condition to check is how fast the error is decreasing using the norm of error from x(n+1) - xstar
 # this depends on the assumption that we are given the root ahead of time before solving so this method
 # is only helpful for checking convergence rates - not actual root finding


import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([1,0])
    
    Nmax = 100
    tol = 1e-10

    guess = [1.5,0] #needed for new stopping criteria
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  SlackerNewton(x0,tol,Nmax,guess)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0]-x[1])

    return F
    
def evalJ(x): 

    
    J = np.array([[6*x[0], -2*x[1]],[6*x[0]*x[1]-3*x[0]**2, 6*x[0]*x[1]]])
    return J


def SlackerNewton(x0,tol,Nmax,guess):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    # initialize list and counter
    i = 0 
    lst = []
    for its in range(Nmax):
       #append x0
       lst.append(evalF(x0))

        # if the norm of the difference between x0 and where the root is greater than 1e-2, recompute J
       if norm(x0 - guess) > 1e-2:  #new recomputing condition
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        # add to counter i so we can keep track of index in lst where we've recomputed jacobian
        i += 1
        #if our difference is small enough, calculate the jacobian at the last x0 and this value won't change
       else:
        permanent_x0 = Jinv = inv(evalJ(lst[i]))

       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its,lst]
           
       x0 = x1
    xstar = x1
    ier = 1
    return[xstar,ier,its,lst]
              
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       


#This method does not perform well at all because there are singular matrix errors, and each of my teammates were able
# to get results for their methods (one computed distance between iterates and the other is checking the ratio of error between iterates)

# It does not perform for the example from class at all since it errors



