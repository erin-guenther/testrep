#3.1.2: Newton and lagrange work better than the monomial - this is probably because of the fact that the inverse of the matrix
#can be hard to calculate when the x-values are super close

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 100
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    #xint = np.linspace(a,b,N+1)

    ''' 3.2 - improve approximation'''
    xint = np.zeros(N+1)
    for j in range(1,N+1):
        xint[j] = np.cos(((2*j-1)*np.pi)/(2*N))

    #the absolute error is much worse at the end points than the uniform nodes
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1001
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
    yeval_bary = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
       yeval_bary[kk] =  evalBarycentric(xeval[kk], xint,yint, N)

    ''' create vector with exact values'''
    fex = f(xeval)


    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval, yeval_bary)
    plt.title("Function values")
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_bary = abs(yeval_bary - fex)
    plt.semilogy(xeval, err_bary, label = "Barycentric")
    plt.legend()
    plt.title("Errors")
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
    

''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

def evalBarycentric(xeval,xint,yint,N):
   
   phi = 1
   #calculate phi to multiply the overall sum by
   for count in range(N+1):
      phi = phi * (xeval - xint[count])

   sum = 0
   #calculate the sum
   for count in range(N+1):
       product = 1
       # calculate the wj denominator
       for jj in range(N+1):
          if (count != jj):
             product = product * (xint[count] - xint[jj])

       # if we're at a node, return the y value so we don't break the code
       if (xint[count] == xeval):
          yeval = yint[count]
          return yeval
       
       # if we're not at a node, follow the 1st formula given on the HW
       else:
          sum += (1/product) * (1/(xeval - xint[count])) * yint[count]
          
   # multiply sum and product together for final approximation
   yeval = sum*phi
   return(yeval)

driver()        
