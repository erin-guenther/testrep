# import libraries
import numpy as np
        
#(1) we can implement the condition that |g'(p)| < 1 to prove the existence of a unique root

def driver():
   f = lambda x: np.exp(x**2 + 7*x - 30) - 1
   fp = lambda x: (2*x+7)*(np.exp(x**2 + 7*x - 30))
   [a,b] = (2,4.5)
   p0 = 4.5
   tol = 1e-10
   Nmax = 1000

   [astar,ier,count] = bisection(f,a,b,tol)
   [pstar,ier,it] = newton(f,fp,p0, tol,Nmax)
   [d,ier, count2] = frankenstein(f,fp,a,b,tol,Nmax)

   print("Bisection:", astar, "it:", count)
   print("Newton:", pstar, "it:", it )
   print("Frankenstein:", d, "it:", count2)


def frankenstein(f,fp,a,b,tol, Nmax):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#     NEW! fp = derivative of f
#     NEW!  Nmax = max iterations
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
      ier = 0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      
      if (abs(fp(d)) < 1) and ((d-f(d)/fp(d)) > a) and (d-f(d)/fp(d) < b):
        ier = 0
        for it in range(Nmax-count):
            p1 = d - f(d)/fp(d)
            if (abs(p1-d) < tol):
                pstar = p1
                info = 0
                return [pstar,info, it+count]
        d = p1
        pstar = p1
        info = 1
        return [pstar,info, it+count]
      
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1

    astar = d
    ier = 0
    return [astar, ier, count]

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
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [pstar,info,it]

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
       return [astar, ier,0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier,0]

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
        return [astar, ier,count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
      
    astar = d
    ier = 0
    return [astar, ier,count]

driver()