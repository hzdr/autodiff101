import numpy as np



def f(x):
    if np.sum(x) > 1.234:
        tmp = np.log(np.power(np.sin(x), 2))
    else:
        tmp = -10 * np.tan(x)
    return np.sum(np.exp(tmp) * 2*np.pi)
