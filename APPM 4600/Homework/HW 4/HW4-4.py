# import libraries
import numpy as np
        
def driver():
  f = lambda x: np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x)
  fp = lambda x: 3*np.exp(3*x)-162*x**5+(108*x**3)*np.exp(x)+(27*x**4)*np.exp(x)-(18*x)*np.exp(2*x)-18*x**2*np.exp(2*x)
  fpp = lambda x: 9*np.exp(3*x) - 810*x**4 + (324*x**2)*np.exp(x) + (108*x**3)*np.exp(x) + (108*x**3)*np.exp(x) + (27*x**4)*np.exp(x) + (-36*x**2)*np.exp(2*x) -72*x*np.exp(2*x) +(-18)*(np.exp(2*x))
  
  g = lambda x: (np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x))/(3*np.exp(3*x)-162*x**5+(108*x**3)*np.exp(x)+(27*x**4)*np.exp(x)-(18*x)*np.exp(2*x)-18*x**2*np.exp(2*x))
  gp = lambda x: 1- ((np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x))*(9*np.exp(3*x) - 810*x**4 + (324*x**2)*np.exp(x) + (108*x**3)*np.exp(x) + (108*x**3)*np.exp(x) + (27*x**4)*np.exp(x) + (-36*x**2)*np.exp(2*x) + (-72*x)*np.exp(2*x) +(-18)*(np.exp(2*x)))/(3*np.exp(3*x)-162*x**5+(108*x**3)*np.exp(x)+(27*x**4)*np.exp(x)-(18*x)*np.exp(2*x)-18*x**2*np.exp(2*x))**2)

  h2 = lambda x: x - 2*(np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x))/(3*np.exp(3*x)-162*x**5+(108*x**3)*np.exp(x)+(27*x**4)*np.exp(x)-(18*x)*np.exp(2*x)-18*x**2*np.exp(2*x))
  h2p = lambda x:  -1 + 2*((np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x))*(9*np.exp(3*x) - 810*x**4 + (324*x**2)*np.exp(x) + (108*x**3)*np.exp(x) + (108*x**3)*np.exp(x) + (27*x**4)*np.exp(x) + (-36*x**2)*np.exp(2*x) + (-72*x)*np.exp(2*x) +(-18)*(np.exp(2*x)))/(3*np.exp(3*x)-162*x**5+(108*x**3)*np.exp(x)+(27*x**4)*np.exp(x)-(18*x)*np.exp(2*x)-18*x**2*np.exp(2*x))**2)

  p0 = 3.5
  Nmax = 1000
  tol = 1.e-5
  
  print('\n')
  print("Newton's method:")
  (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('Number of iterations:', '%d' % it)
  print('\n')

  print("Using modified Newton's from class")
  (p,pstar,info,it) = newton(g,gp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('Number of iterations:', '%d' % it)

  print('\n')
  print("Using modified Newton's from 2(c)")
  (p,pstar,info,it) = newton(h2,h2p,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('Number of iterations:', '%d' % it)
  print('\n')



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
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
            
driver()