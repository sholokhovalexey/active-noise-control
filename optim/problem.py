import math
import numpy as np
from copy import deepcopy
from scipy.optimize import Bounds, NonlinearConstraint
from collections import OrderedDict

from common.general import compose
from optim.utils import default_bounds, unpack_bounds


def name_generator(dtype=str):
    i = 0
    while True:
        i += 1
        yield dtype(i)
        
        
class Problem:
    def __init__(self, dim=None):
        # self.parametrization = lambda x: x
        self.dim = dim
        self.bounds = Bounds()
        self.constraints = OrderedDict()
        self.counter = 0
        self.is_unconstrained = False
        
    def set_parametrization(self, parametrization=None):
        if parametrization is None:
            parametrization = lambda x: x
        self.parametrization = parametrization
        
    def add_constraint(self, func, name=None, keep_feasible=True):
        name = f"constraint_{self.counter}" if name is None else name
        self.counter += 1
        assert not name in self.constraints, f"'{name}' already exsits!"
        lb, ub = (-np.inf, 0.000)
        #NOTE: NonlinearConstraint supports vector-valued (func, lb, ub)
        
        # f = compose(func, self.parametrization)
        f = func
        self.constraints[name] = NonlinearConstraint(
            f, 
            lb, 
            ub, 
            jac="3-point", 
            keep_feasible=keep_feasible,
        )
        
    def violation(self, x):
        for (name, constraint) in self.constraints.items():
            val = constraint.fun(x)
            is_violated = val > 0
            sign = "!<" if is_violated else "<"
            msg = f"{'[X]' if is_violated else '[OK]'}:\t{val} \t{sign} 0 ({name})"
            print(msg)
        
    def get_constraints(self):
        constraints = {}
        for (name, constraint) in self.constraints.items():
            f = compose(constraint.fun, self.parametrization)
            constraints[name] = f
        return constraints
        
    def add_bounds(self, *args, keep_feasible=True):
        if args[0] is None:
            self.bounds = default_bounds()
        elif isinstance(args[0], Bounds):
            self.bounds = args[0]
        else:
            if len(args) == 1:
                lb, ub = unpack_bounds(args[0])
            elif len(args) == 2:
                lb, ub = args
            self.bounds = Bounds(lb, ub, keep_feasible=keep_feasible)
        
    def set_bounds(self, bounds):    
        assert isinstance(bounds, Bounds)
        self.bounds = bounds
        
    def to_unconstrained(self, weights):
        if isinstance(weights, dict):
            assert set(self.constraints.keys()) == set(weights.keys())
        else:
            assert len(self.constraints) == len(weights)
            weights = OrderedDict()
            for (k, w) in zip(self.constraints.keys(), weights):
                weights[k] = w
        problem = deepcopy(self)
        problem.weights = weights
        problem.is_unconstrained = True
        return problem
        
    def objective(self, x):
        raise NotImplementedError
    
    def __call__(self, x_raw):
        x = self.parametrization(x_raw)
        
        loss = self.objective(x)
        
        if self.is_unconstrained:
            for (name, constraint) in self.constraints.items():
                weight = self.weights[name]
                eps = 1e-300
                log_barrier_fn = lambda u: (-1) * np.log(np.maximum(eps, -u))
                loss += weight * log_barrier_fn(constraint.fun(x))
        return loss
    
