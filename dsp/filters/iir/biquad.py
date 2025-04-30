import math
import numpy as np

from numpy import sin, cos, sqrt, exp


def parametric_eq(
    cutoff_freq,
    q_factor,
    gain_db,
    sample_rate,
    filter_type,
):
    if isinstance(cutoff_freq, (np.ndarray)):
        assert cutoff_freq.shape == q_factor.shape == gain_db.shape
        shape = cutoff_freq.shape

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = sin(w0) / (2 * q_factor)
    cos_w0 = cos(w0)
    sqrt_A = sqrt(A)
    
    c0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    c1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    c2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    # c0 + c1 + c2 = 4 * (1 - cos_w0)
    # c0 - c1 + c2 =
    
    d0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    d1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    d2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    # d0 + d1 + d2 = A * 4 * (1 - cos_w0)
    # d0 - d1 + d2 = 

    if filter_type in ["high_shelf", "high shelf", "hshelf", "hs"]:
        b0 = A * c0
        b1 = A * c1
        b2 = A * c2
        a0 = d0
        a1 = d1
        a2 = d2
    elif filter_type in ["low_shelf", "low shelf", "lshelf", "ls"]:
        b0 = A * d0
        b1 = A * d1
        b2 = A * d2
        a0 = c0
        a1 = c1
        a2 = c2
    elif filter_type in ["peaking", "peak", "pk"]:
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type in ["low_pass", "low pass", "lpass", "lp"]:
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type in ["high_pass", "high pass", "hpass", "hp"]:
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    sos = np.asarray([b0, b1, b2, a0, a1, a2])
    sos = sos / a0
    return sos


def sos_general_parametric(
    f, 
    Q,
    g,
    fs, 
    parametrization="vv",
    ):
    
    w0 = 2 * math.pi * (f / fs)
    cos_w0 = cos(w0)
    
    if parametrization == "vv":
        
        alpha = exp(-Q * w0)
        c0 = 1
        c1 = -2 * cos_w0 * alpha
        c2 = alpha ** 2
        
    elif parametrization == "as":
        
        alpha = sin(w0) / (2 * Q)
        c0 = 1 + alpha
        c1 = -2 * cos_w0
        c2 = 1 - alpha
        
    scale = 10 ** (g / 20.0)
    c0, c1, c2 = scale * c0, scale * c1, scale * c2 
    
    return c0, c1, c2