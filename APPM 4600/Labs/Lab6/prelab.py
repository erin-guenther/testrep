import numpy as np

# f'(s) = f(s+h) - f(s) / h
# f'(s) = f(s+h) - f(s-h) / 2h

def driver(): #has order of 1
    f = forward()
    c = centered()

    print("Forward derivative:", f)
    print("Centered derivative:", c)

    print("error 1:", abs(np.array(f) + 1))
    print("error 2:", abs(np.array(c) + 1))



def forward(): #has order of 1
    h = 0.01 * 2. ** (-np.arange(0,10))

    f = lambda x: np.cos(x)

    l = []
    for i in h:
        y = (f(np.pi/2 + i) - f(np.pi/2))/i
        l.append(y)

    return l

def centered():
    h = 0.01 * 2. ** (-np.arange(0,10))

    f = lambda x: np.cos(x)

    l = []
    for i in h:
        y = (f(np.pi/2 + i) - f(np.pi/2 - i))/(2*i)
        l.append(y)

    return l

driver()