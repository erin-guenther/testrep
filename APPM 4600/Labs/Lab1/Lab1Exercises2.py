import numpy as np
import numpy.linalg as la
import math


def driver():
    n = 2
    x = np.linspace(0,np.pi,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    
    """
    f = lambda x: np.sin(x)
    g = lambda x: np.cos(x)
    y = f(x)
    w = g(x)
    """
    y = np.array([[1,2],[1,2]])
    w = np.array([[2,1],[2,1]])
    # evaluate the dot product of y and w
    # dp = dotProduct(y,w,n)
    product = matrixMultiplication(y,w,n)
    # print the output
    # print('the dot product is :', dp)
    print('the product of the matrices is:', product)
    return 

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

def matrixMultiplication(x,y,n):
    product = np.zeros((n,n))
    # do dot product for row x column transposed?
    for i in range(n):
        for j in range(n):
            xRow = x[i]
            yRow = np.transpose(y[:,j])
            product[i,j] = dotProduct(xRow,yRow,n)

    return product


driver()