import numpy as np
from tqdm import tqdm
from dsp.filters import xfilt, init_zi


def anc_simulation(x, d, FF, FB, SP, verbose=False):
    
    zi_sp = init_zi(SP)
    zi_fb = init_zi(FB)
    
    # FF
    if x is not None and FF is not None:
        x, _ = xfilt(FF, x)

    if x is None:
        nIters = len(d)
    else:
        nIters = min(len(x), len(d))
        
    y = np.zeros(nIters)
    
    # zero = np.zeros(1)
    
    ff_n = 0
    fb_n = 0
    fffb_n = 0
    
    if verbose:
        iters = tqdm(range(nIters))
    else:
        iters = range(nIters)
    
    for n in iters:
        
        if x is not None:
            ff_n = x[n]
        else:
            ff_n = 0
            
        fffb_n = ff_n + fb_n
        
        assert np.isfinite(fffb_n)
        
        # SP
        y_n, zi_sp = xfilt(SP, fffb_n, zi=zi_sp)
        # residual
        e_n = d[n] - y_n
        # FB
        fb_n, zi_fb = xfilt(FB, e_n, zi=zi_fb)
        
        y[n] = y_n
        
    return y
        

    