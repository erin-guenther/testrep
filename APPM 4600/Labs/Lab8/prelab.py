import numpy as np

def fint_int(xeval,xint):
    sub_ints = []
    for i in range(len(xint)):
        sub_ints.append(np.where(xeval <= xint[i]))
    return sub_ints

def eval_linspline(x,x0,x1,f):
    m = (f(x1) - f(x0))/(x1-x0)
    return m*(x-x0)+f(x0)

