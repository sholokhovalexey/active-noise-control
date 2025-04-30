import math
import numpy as np


def compose(*args):
    def func(x):
        for f in args[::-1]:
            x = f(x)
        return x
    return func





