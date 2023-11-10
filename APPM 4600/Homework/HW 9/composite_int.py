import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1/(1+x**2)
    a = -5
    b = 5
    
    # exact integral
    I_ex = 2.746801533890032

# for simpson's n must be even.        
# n+1 = number of pts.
    nT = 913
    nS = 107
    I_trap = CompTrap(a,b,nT,f)
    print('I_trap= ', I_trap)
    
    err = abs(I_ex-I_trap)   
    
    print('absolute error = ', err)    
    
    I_simp = CompSimp(a,b,nS,f)

    print('I_simp= ', I_simp)
    
    err = abs(I_ex-I_simp)   
    
    print('absolute error = ', err)    

    [I_quad, abs2] = scipy.integrate.quad(f, a,b, epsrel = 1e-4)

    print('Scipy quadrature with 1e-4 tol:', I_quad)
    print('absolute error:', abs2)

    [I_quad2, abs2] = scipy.integrate.quad(f, a,b, epsrel = 1e-6)

    print('Scipy quadrature with 1e-6 tol:', I_quad2)
    print('absolute error', abs2)



        
def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    
    return I_trap     

def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # even part 
         I_simp = I_simp+2*f(xnode[2*j])
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp
    
    return I_simp     


    
    
driver()    
