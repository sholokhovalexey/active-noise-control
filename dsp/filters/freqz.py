import math
import numpy as np
from common.math import logscale
from common.math import polyval


def freqz(
    b,
    a,
    worN,
    whole=False,
    fs=2 * np.pi,
    log=False,
    include_nyquist=False,
    eps=1e-16,
    mode="poly",
):
    """Compute the frequency response of a digital filter."""

    assert mode in ["poly", "fft"]

    # w
    if fs is not None:
        lastpoint = 2 * np.pi if whole else np.pi
        if log:
            w = logscale(0, lastpoint, worN, endpoint=include_nyquist and not whole)
        else:
            w = np.linspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)
    else:
        w = None

    # h
    if mode == "fft":
        B = np.fft.rfft(b, n=worN * 2)[..., :worN]
        A = np.fft.rfft(a, n=worN * 2)[..., :worN]
        h = (B / (A + eps)) + eps

    else:
        zm1 = np.exp(-1j * w)
        if isinstance(b, (int, float)) or b.ndim == 0:
            h *= b
        else:
            b = np.flip(b, -1)
            h = polyval(b, zm1) 
            
        if isinstance(a, (int, float)) or a.ndim == 0:
            h /= a
        else:
            a = np.flip(a, -1)
            h /= (polyval(a, zm1) + eps)
    
    if fs is not None:
        w = w * fs / (2 * np.pi)

    return w, h


def sosfreqz(sos, worN=512, whole=False, fs=2 * np.pi, log=False, mode="poly"):
    """Compute the frequency response of a digital filter in SOS format."""

    h = 1.0
    for row in sos:
        w, rowh = freqz(
            row[:3], 
            row[3:], 
            worN=worN, 
            whole=whole, 
            fs=fs, 
            log=log,
            mode=mode,
        )
        h *= rowh
 
    return w, h
