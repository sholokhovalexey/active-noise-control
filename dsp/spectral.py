import numpy as np
from scipy.signal import welch


def attenuation(x, y, **kwargs):
    # see scipy.signal.welch for details
    f, Pxx = welch(x, **kwargs)
    f, Pyy = welch(y, **kwargs)
    eps = 1e-14
    att = 10 * np.log10(Pxx + eps) - 10 * np.log10(Pyy + eps)
    return f, att


def rel_diff_db(X, Y, weighting=None, start=0, stop=-1):
    """calc log10(|1 - X/Y|^2)"""
    d = np.abs(1 - X * Y.conj() / (Y * Y.conj() + 1e-16))**2
    d = np.log10(d + 1e-18)
    if weighting is not None:
        d = d[:len(weighting)] * weighting
    else:
        if stop < 0:
            stop = len(d) + stop
        d = d[start:stop] # TODO: frequency band of interest, or OCTAVE bands
    return np.sum(d)

