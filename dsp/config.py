from common.math import next_power_of_two


def get_spectral_args(fs, window="hann", win=None, hop=None, nfft=None):
    
    win = int(next_power_of_two(fs // 12)) if win is None else win
    hop = win // 2 if hop is None else hop
    nfft = win if nfft is None else nfft
    assert win <= nfft
    
    spectral_args = {
        "fs": fs,
        "window": window, 
        "nperseg": win, 
        "noverlap": hop, 
        "nfft": nfft, 
        "detrend": False, 
        "return_onesided": True, 
        "scaling": "spectrum",
    }
    return spectral_args
    
    
# default
fs = 192000
spectral_args = get_spectral_args(fs)

    