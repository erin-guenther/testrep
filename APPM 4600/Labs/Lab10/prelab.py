import numpy as np

def driver():
  x= 1
  lst = eval_legendre(1,3)
  lst

def eval_legendre(x,n):
    if (n==0):
       return [1]
    elif (n==1):
       return [1,x]
    else:
       lst = np.zeros(n+2)
       lst[0] = 1
       lst[1] = x
       for i in range(2,n+2):
          lst[i] = 1 / (n+1) * ((2*n+1)*x*lst[i-1] - n*lst[i-2])
    
    return lst
