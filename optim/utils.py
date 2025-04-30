from scipy.optimize import Bounds


def default_bounds(dim=None):
    if dim is None:
        return Bounds(keep_feasible=True)
    else:
        lb = [-np.inf] * dim
        ub = [np.inf] * dim
        return Bounds(lb, ub, keep_feasible=True)
    
    
def unpack_bounds(bounds):
    if isinstance(bounds, Bounds):
        lb, ub = bounds.lb, bounds.ub
    else:
        if len(bounds) == 2 and len(bounds[0]) != 2:
            lb, ub = bounds
        else:
            lb, ub = list(zip(*bounds))
    return lb, ub