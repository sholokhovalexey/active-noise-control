import math
import numpy as np
import torch
import torch.nn.functional as F

from dsp.ddsp.spectral import polyval
from ..fir import freqdomain_fir

# TODO: w, h = ...(), w is returned as np.array --> torch ?


def _apply_freqdomain_fir(x, H, n_fft):

    # add extra dims to broadcast filter across
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # apply as a FIR filter in the frequency domain
    y = freqdomain_fir(x, H, n_fft)

    # crop
    y = y[..., : x.shape[-1]]

    return y



def lfilter_via_fsm(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor = None):
    """Use the frequency sampling method to approximate an IIR filter.
    The filter will be applied along the final dimension of x.
    Args:
        x (torch.Tensor): Time domain signal with shape (bs, 1, timesteps)
        b (torch.Tensor): Numerator coefficients with shape (bs, N).
        a (torch.Tensor): Denominator coefficients with shape (bs, N).
    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, 1, timesteps)
    """
    bs, chs, seq_len = x.size()  # enforce shape
    assert chs == 1

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # move coefficients to same device as x
    b = b.type_as(x)

    if a is None:
        # directly compute FFT of numerator coefficients
        H = torch.fft.rfft(b, n_fft)
    else:
        a = a.type_as(x)
        # compute complex response as ratio of polynomials
        H = fft_freqz(b, a, n_fft=n_fft)

    return _apply_freqdomain_fir(x, H, n_fft)


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
            w = np.logspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)
        else:
            w = np.linspace(0, lastpoint, worN, endpoint=include_nyquist and not whole)
        w = torch.tensor(w, device=b.device)
    else:
        w = None

    # h
    if mode == "fft":
        
        B = torch.fft.rfft(b, n=worN * 2)[..., :worN]
        A = torch.fft.rfft(a, n=worN * 2)[..., :worN]
        h = (B / (A + eps)) + eps

    else:
        
        h = None

        if isinstance(a, (int, float)) or a.ndim == 0 or a.numel() == 1:
            n_fft = worN if whole else worN * 2
            h = torch.fft.rfft(b, n=n_fft)[..., :worN]
            h /= a

        if h is None:
            zm1 = torch.exp(-1j * w)
            b = torch.flip(b, [-1])
            a = torch.flip(a, [-1])
            h = polyval(b, zm1) / (polyval(a, zm1) + eps)
            # B = torch.fft.rfft(b, n=worN * 2)[..., :worN]

        # TODO: need to catch NaNs here
    
    if fs is not None:
        w = w * fs / (2 * np.pi)

    return w, h


def fft_freqz(b, a, n_fft):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H