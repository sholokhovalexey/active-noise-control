import math
import numpy as np

from common.general import compose
from optim.utils import default_bounds, unpack_bounds
from optim.solver.base import Solver

from mealpy import FloatVar, Problem
from mealpy import ABC, PSO, DE, GWO


class ProblemWrapperMealpy(Problem):
    def __init__(self, obj_fn, bounds=None, minmax="min", **kwargs):
        self.obj_fn = obj_fn
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x = self.decode_solution(x)["x"]
        loss = self.obj_fn(x)
        return loss

    
class SolverMetaHeuristic(Solver):
    def __init__(self, problem):
        super().__init__()
        
        bounds = self.get_bounds(problem)
            
        lb, ub = unpack_bounds(bounds)
        lb, ub = list(lb), list(ub)
        bounds = [FloatVar(lb=lb, ub=ub, name="x")]
        
        problem_mealpy = ProblemWrapperMealpy(problem, bounds)
        problem_mealpy.log_to = None
        self.problem = problem_mealpy
        
    def prepare(self, method, epoch=100, pop_size=30, **options):
        self.method = method.lower()
        if self.method in ["abc"]:
            solver = ABC.OriginalABC(epoch=epoch, pop_size=pop_size, n_limits=50)
        elif self.method in ["pso"]:
            solver = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4)
        elif self.method in ["de"]:
            solver = DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.7, cr=0.9, strategy=0)
        elif self.method in ["gwo"]:
            solver = GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
        self.solver = solver
        self.is_initialized = True
        
    def solve(self, x0=None, seed=42, mode="single", n_workers=None):
        assert self.is_initialized
        
        if x0 is not None:
            x0 = [x0] * self.solver.pop_size
        
        self.solver.solve(
            self.problem, 
            mode=mode, 
            n_workers=n_workers,
            termination=None, 
            starting_solutions=x0,
            seed=seed,
        )
        # mode: Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.
        # starting_solutions: List or 2D matrix (numpy array) with length equal pop_size 
        
        sol = self.solver.problem.decode_solution(self.solver.g_best.solution)
        
        # print(f"Best agent: {self.solver.g_best}")
        # print(f"Best solution: {self.solver.g_best.solution}")
        # print(f"Best accuracy: {self.solver.g_best.target.fitness}")
        # print(f"Best parameters: {sol}")
        return sol["x"]
        
