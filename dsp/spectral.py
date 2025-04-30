import numpy as np
from scipy.signal import welch


def attenuation(x, y, **kwargs):
    # see scipy.signal.welch for details
    f, Pxx = welch(x, **kwargs)
    f, Pyy = welch(y, **kwargs)
    eps = 1e-14
    att = 10 * np.log10(Pxx + eps) - 10 * np.log10(Pyy + eps)
    return f, att
    