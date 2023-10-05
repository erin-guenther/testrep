import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver(): #has order of 1
    x = np.array([1,0])
    h = [0.01,0.001,0.0001]
    tol = 1e-10

    [l,ier] = forward(h,x, tol)
    print("Using forward derivative:", l)


def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0]-x[1])

    return F

def forward(h,x,tol): #has order of 1


    l = []
    for i in h:
        F = evalF(x)
        F2 = evalF([x[0],x[1]+i])
        F1 = evalF([x[0]+i, x[1]])

        fx = (F1[0]-F[0])/i
        fy = (F2[0] - F[0])/i
        gx = (F1[0] - F[0])/i
        gy = (F2[0] - F[0])/i
        x = (inv(np.array([[fx,fx],[gx,gy]])))

        x1 = x - x.dot(F)
       
        if (norm(x1-x) < tol):
           xstar = x1
           ier =0
           l.append(xstar)
           
        x0 = x1
    
    xstar = x1
    ier = 1
    return[l,ier]   


driver()