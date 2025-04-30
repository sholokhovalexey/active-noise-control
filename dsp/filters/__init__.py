from .fir import *
from .iir import *
from .freqz import *
from scipy.signal import lfilter, sosfilt


def to_atleast_1d(x):
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(-1)
    return x


def xfilt(coeffs, x, zi=None):
    x = to_atleast_1d(x)
    if coeffs.ndim == 1:
        x = lfilter(coeffs, 1, x, zi=zi)
    elif coeffs.ndim == 2:
        x = sosfilt(coeffs, x, zi=zi)
    if zi is not None:
        x, zi = x
    return x, zi


def init_zi(coeffs):
    if coeffs.ndim == 1:
        zi = np.zeros(len(coeffs)-1,)
    elif coeffs.ndim == 2:
        n_sections, _ = coeffs.shape
        zi = np.zeros((n_sections, 2))
    return zi