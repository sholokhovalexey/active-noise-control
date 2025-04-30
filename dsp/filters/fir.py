import numpy as np
from dsp import conv_causal
# from scipy.signal import lfilter


def firfilt(coeff, x):
    # scipy.signal style args
    return conv_causal(x, coeff)


def load_fir(path):
    ir = np.loadtxt(path)
    return ir