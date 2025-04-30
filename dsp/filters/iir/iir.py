import os
import numpy as np
from scipy.signal import sosfilt
from dsp.core import delta_function


# sos
def load_sos(path):
    sos = np.loadtxt(path, delimiter=",")
    n_sections, n_coeffs = sos.shape
    assert n_coeffs in [5, 6]
    # coeffs: (b0, b1, b2, [1,] a1, a2)
    if n_coeffs == 5:
        # insert a0 = 1
        sos = np.insert(sos, 3, 1.0, axis=-1)
    return sos


def save_sos(path, sos):
    n_sections, n_coeffs = sos.shape
    assert n_coeffs in [5, 6]
    # coeffs: (b0, b1, b2, [1,] a1, a2)
    if n_coeffs == 5:
        # insert a0 = 1
        sos = np.insert(sos, 3, 1.0, axis=-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, sos, fmt="%.12f", delimiter=",")


def sos_to_fir(sos, order):
    delta = delta_function(order)
    ir = sosfilt(sos, delta)
    return ir

