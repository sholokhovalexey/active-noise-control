import math
import numpy as np

from common.general import compose
from optim.utils import default_bounds, unpack_bounds
from optim.solver.base import Solver

from scipy.optimize import minimize
from scipy.optimize import Bounds, NonlinearConstraint


class SolverGradient(Solver):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        
    def prepare(self, method="trust-constr", maxiter=100, **options):
        self.method = method
        self.maxiter = maxiter
        self.options = options
        self.is_initialized = True
    
    def solve(self, x0=None, seed=42):
        assert self.is_initialized
        
        np.random.seed(seed)
        
        bounds = self.get_bounds(self.problem)
        
        constraints = []
        for f in self.problem.get_constraints().values():
            constraints += [NonlinearConstraint(f, -np.inf, 0, jac="3-point", keep_feasible=True)]
        
        funtion = lambda x: self.problem(x)
        
        self.options.update(
            {
                "maxiter": self.maxiter, 
                "verbose": 1, # 2 - per iter
            }
        )
        
        if x0 is None:
            lb = np.array(bounds.lb)
            ub = np.array(bounds.ub)
            x0 = (ub - lb) * np.random.rand() + lb # may be infeasible

        res = minimize(
            funtion, 
            x0,
            method=self.method, 
            bounds=bounds, 
            constraints=constraints,
            options=self.options,
        )
        # print("Success:", res.success)
        print(res)
        
        return res.x

