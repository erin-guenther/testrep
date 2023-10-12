# matrix: copy from notes basically
#matrix has form: V = [[1, x1, x1**2, ..., x1**n-1],
                # [1, x2, x2**2, ..., x2**n-1], ...
                # [1, xm, xm**2, ..., xn**n-1]] for n points in the interval and n-1 degrees
# and use V * [a0,a1,a2,a3,a4]^T

#plan: solve this 
# (something along the lines of) px = b[0] + b[1]x + b[2]x**2 (but use a for loop since we dont know the degree for each problem)

# we'll be exploring the interpolation code and changing the values of the nodes to see how it impacts the accuracy

