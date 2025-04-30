import math
import numpy as np
from optim.utils import default_bounds, unpack_bounds


class Solver:
    def __init__(self):
        self.is_initialized = False
        
    @staticmethod
    def get_bounds(problem):
        bounds = None
        if hasattr(problem, "bounds"):
            bounds = problem.bounds
        elif problem.dim is not None:
            bounds = default_bounds(problem.dim)
        else:
            raise ValueError("Problem should have attribute 'bounds'")
        return bounds
    
    def solve(self):
        pass