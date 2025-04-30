import math
import numpy as np


def polyval(p, x):
    """Evaluate a polynomial at specific values.
    Args:
        p (coeffs), [a_{N}, a_{N-1}, ..., a_1, a_0]
        x (values)

    Returns:
        v (values)
    """
    N = len(p)
    val = 0
    for i in range(N - 1):
        val = (val + p[i]) * x
    return (val + p[-1])


def logscale(start, stop, num=100, base=10.0, endpoint=True):
    t = np.logspace(start, stop, num=num, base=base, endpoint=endpoint) # [base**start, base**stop]
    t = (t - t[0]) / (t[-1] - t[0]) * (stop - start) + start
    return t

    
def next_power_of_two(x):
    power = np.ceil(np.log2(x)) #TODO: x==0, of negative
    return 2**power


def relative_difference(x, y, eps=1e-16):
    return np.linalg.norm(x - y) / (np.linalg.norm.norm(y) + eps)