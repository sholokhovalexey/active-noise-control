import numpy as np
from tqdm import tqdm

from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft
from dsp.filters import freqz, sosfreqz
from scipy.signal import sosfilt


# def fdaf_updater(H, mu=0.5, beta=0.1):
#     norm = np.full(H.size, 1e-8)
#     buffer = []

#     def updater(H, X_n, E_n, update=False):
#         nonlocal norm, buffer, mu, beta
#         norm = beta * norm + (1 - beta) * np.abs(X_n)**2
#         G = mu * E_n / (norm + 1e-4)
#         H_step = X_n.conj() * G
#         buffer.append(H_step)
#         if update:
#             H = H + np.mean(buffer, axis=0)
#             buffer = []
#             return H
#     return updater


def fdaf_updater(H, mu=1, beta=0.1):
    norm = np.full(H.size, 1e-8)
    buffer = []

    def updater(H, X_n, E_n, update=False):
        nonlocal norm, buffer, mu, beta
        norm = beta * norm + (1 - beta) * np.abs(X_n)**2
        # H_step = mu * E_n / (X_n + 1e-4) # naive, unstable
        # H_step = mu * E_n / (np.sqrt(norm) * np.exp(1j * np.angle(X_n)) + 1e-4)
        H_step = mu * E_n * X_n.conj() / (norm + 1e-4) # better
        buffer.append(H_step)
        if update:
            H = H + np.mean(buffer, axis=0)
            buffer = []
            return H
    return updater
    

def fdaf(
    x,
    d, 
    frame_size,
    filter_obj,
    params_init,
    mu=0.5,
    beta=0.1,
    update_interval=1, 
):
    
    # init filter
    params = params_init.copy()
    f, H = filter_obj.freqz(params)
    
    # update rule
    updater = fdaf_updater(H, mu=mu, beta=beta)
    
    window = np.hanning(frame_size)
    x_old = np.zeros(frame_size)
    zeros = np.zeros(frame_size)
    
    num_block = min(len(x), len(d)) // frame_size

    y = np.zeros_like(d)
    for n in tqdm(range(num_block)):

        mask = slice(n*frame_size, (n+1)*frame_size)
        
        # current input frame
        x_n = np.r_[x_old, x[mask]]
        X_n = fft(x_n)
        x_old = x[mask]

        # filtering
        y_n = filter_obj.apply(params, x_n)
        y_n = y_n[frame_size:]
        y[mask] = y_n

        # error frame
        d_n = d[mask]
        e_n = d_n - y_n
        e_fft = np.r_[zeros, e_n*window]
        E_n = fft(e_fft)

        # update in freq domain 
        if (n + 1) % update_interval == 0:
            H = updater(H, X_n, E_n, update=True)
            
            h = ifft(H)
            h[frame_size:] = 0
            H = fft(h)
        else:
            updater(H, X_n, E_n, update=False)

        # update filter params
        params = filter_obj.fit_freqz(params, H)
        f, H = filter_obj.freqz(params)
                
    return y, params
