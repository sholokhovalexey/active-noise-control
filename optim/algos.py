import numpy as np
from tqdm import tqdm


def calc_step(x, loss_fn, cfg):
    
    x_step = np.zeros_like(x)
    loss_current = loss_fn(x)
    
    for i in range(len(x)):
        x_new = x.copy()

        step = cfg.step[i]
        low, up = cfg.bounds[i]

        step_pos = min(x[i] + step, up) - x[i]
        step_neg = max(low, x[i] - step) - x[i]
        
        x_new[i] = x[i] + step_pos

        loss_new = loss_fn(x_new)
        if loss_new < loss_current:
            x_step[i] = step_pos
        else:
            x_step[i] = step_neg
    return x_step


def step_fixed(x, loss_fn, cfg):

    x_step = calc_step(x, loss_fn, cfg)

    loss_current = loss_fn(x)
        
    step_size = cfg.step_size_max
    for i in range(cfg.n_steps_linesearch):
    
        x_new = x + step_size * x_step
        
        loss_new = loss_fn(x_new)
    
        if loss_new < loss_current:
            loss_current = loss_new
            x = x_new
            break
    
        step_size = step_size / 2

    return x


def step_fixed_coord(x, loss_fn, cfg):
    
    for i in range(len(x)):
        x_new = x.copy()

        loss_current = loss_fn(x)

        step = cfg.step[i]
        low, up = cfg.bounds[i]

        step_pos = min(x[i] + step, up) - x[i]
        step_neg = max(low, x[i] - step) - x[i]
        
        x_new[i] = x[i] + step_pos

        loss_new = loss_fn(x_new)
        
        if loss_new < loss_current:
            x[i] = x[i] + step_pos
        elif loss_new > loss_current:
            x[i] = x[i] + step_neg
            
    return x    


