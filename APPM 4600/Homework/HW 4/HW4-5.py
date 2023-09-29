import numpy as np
import math
import matplotlib.pyplot as plt

def driver():
    p0 = 2
    p1 = 1
    f = lambda x: x**6 - x -1
    fp = lambda x: 6*x**5 - 1
    tol = 1e-12
    Nmax = 10000

    [pstar,ier,lst] = secant_list(p0,p1,f,tol,Nmax)

    error  = np.array(lst)
    error = error - pstar
    print("Secant method:")
    print('{:<10} {:<16}'.format('Iteration', 'Error'))
    for i in range(len(error)):
        print('{:<10} {:<16}'.format(i , error[i]))

    print('')

    print("Newton's method:")
    (p,pstar2,info,it,lst2) = newton(f,fp,p0,tol, Nmax)
    error2 = np.array(lst2)
    error2 = error2 - pstar2

    print('{:<10} {:<16}'.format('Iteration', 'Error'))
    for i in range(len(error2)):
        print('{:<10} {:<16}'.format(i , error2[i]))

    x = abs(error2[1:-1])
    y = abs(error2[:-2])
    print(x,y)

    plt.plot(np.log(y),np.log(x))
    plt.show()


def secant_list(p0,p1,f,tol,Nmax):
    lst = []
    if (f(p0)==0 or f(p1)==0):
        pstar = p0
        ier = 1
        return [pstar,ier,lst]
    
    fp1 = f(p1)
    fp0 = f(p0)

    for i in range(Nmax):
        if (abs(fp1 - fp0)==0):
            ier = 1
            pstar = p1
            return [pstar,ier,lst]
        
        p2 = p1 - fp1*(p1-p0)/(fp1-fp0)

        if (abs(p2-p1) < tol):
            pstar = p2
            ier = 0
            lst.append(p1)
            lst.append(p2)
            return [pstar,ier,lst]
        
        lst.append(p0)
        p0 = p1
        fp0 = fp1
        
        p1 = p2
        fp1 = f(p2)

    
    pstar = p2
    ier = 1
    return [pstar,ier,lst]


def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  lst = []
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          lst.append(p1)
          return [p,pstar,info,it,lst]
      lst.append(p0)
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it,lst]
    
driver()