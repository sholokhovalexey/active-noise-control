import math
import numpy as np
import torch
import torch.nn.functional as F

from common.math import sin, cos, sqrt, exp
from ..fir import freqdomain_fir
from .iir import fft_freqz, freqz
from .iir import _apply_freqdomain_fir


def _validate_sos(sos, eps=1e-8, normalize=False):
    """Helper to validate a SOS input."""

    if sos.ndim == 2:
        n_sections, m = sos.shape
        bs = 0
    elif sos.ndim == 3:
        bs, n_sections, m = sos.shape
        # flatten batch into sections dim
        sos = sos.reshape(bs * n_sections, 6)
    else:
        raise ValueError(
            "sos array must be shape (batch, n_sections, 6) or (n_sections, 6)"
        )

    if m != 6:
        raise ValueError(
            "sos array must be shape (batch, n_sections, 6) or (n_sections, 6)"
        )

    # remove zero padded sos
    # sos = sos[sos.sum(-1) != 0,:]

    # normalize by a0
    if normalize:
        a0 = sos[:, 3].unsqueeze(-1)
        sos = sos / a0

    # if not (sos[:, 3] == 1).all():
    #    raise ValueError('sos[:, 3] should be all ones')

    # fold sections back into batch dim
    if bs > 0:
        sos = sos.view(bs, -1, 6)
    return sos, bs, n_sections


# TODO: batch version (filter_type is a tensor of str)
def parametric_eq(
    cutoff_freq,
    q_factor,
    gain_db,
    sample_rate,
    filter_type,
):
    if isinstance(cutoff_freq, (torch.Tensor, np.ndarray)):
        assert cutoff_freq.shape == q_factor.shape == gain_db.shape
        shape = cutoff_freq.shape

    # reshape params
    if isinstance(cutoff_freq, torch.Tensor):
        gain_db = gain_db.view(-1)
        cutoff_freq = cutoff_freq.view(-1)
        q_factor = q_factor.view(-1)

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

    # b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    # a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)
    # b = b.type_as(gain_db) / a0
    # a = a.type_as(gain_db) / a0

    if isinstance(cutoff_freq, torch.Tensor):
        sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=1)  # (-1, 6)
        sos = sos.type_as(gain_db) / a0.view(-1, 1)
        sos = sos.reshape(*shape, 6)
    # elif isinstance(cutoff_freq, np.ndarray):
    else:
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


def sosfilt_via_fsm(sos: torch.Tensor, x: torch.Tensor):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    The filter will be applied along the final dimension of x.
    Args:
        sos (torch.Tensor): Tensor of coefficients with shape (bs, n_sections, 6).
        x (torch.Tensor): Time domain signal with shape (bs, ... , timesteps)

    Returns:
        y (torch.Tensor): Filtered time domain signal with shape (bs, ..., timesteps)
    """
    bs = x.size(0)

    # round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # compute complex response as ratio of polynomials
    H = fft_sosfreqz(sos, n_fft=n_fft)

    return _apply_freqdomain_fir(x, H, n_fft)


def fft_sosfreqz(sos, n_fft=512):
    """Compute the complex frequency response via FFT of cascade of biquads

    Args:
        sos (torch.Tensor): Second order filter sections with shape (bs, n_sections, 6)
        n_fft (int): FFT size. Default: 512
    Returns:
        H (torch.Tensor): Overall complex frequency response with shape (bs, n_bins)
    """
    n_sections, n_coeffs = sos.shape[-2:]
    assert n_coeffs == 6  # must be second order
    for section_idx in range(n_sections):
        b = sos[..., section_idx, :3]
        a = sos[..., section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H


def sosfreqz(sos, worN=512, whole=False, fs=2 * np.pi, log=False, fast=False):
    """Compute the frequency response of a digital filter in SOS format.

    Args:
        sos (Tensor): Array of second-order filter coefficients, with shape
        (n_sections, 6) or (batch, n_sections, 6).

    Returns:
        w (array): freqs
        h (Tensor): frequency response
    """

    sos, bs, n_sections = _validate_sos(sos)

    if n_sections == 0:
        raise ValueError("Cannot compute frequencies with no sections")
    h = 1.0

    # check for batches (if none add batch dim)
    if bs == 0:
        sos = sos.unsqueeze(0)

    # this method of looping over SOS is somewhat slow
    if not fast:
        for row in torch.chunk(sos, n_sections, dim=1):
            # remove batch elements that are NaN
            row = torch.nan_to_num(row)
            row = row.reshape(-1, 6)  # shape: (batch_dim, 6)
            w, rowh = freqz(
                row[:, :3], 
                row[:, 3:], 
                worN=worN, 
                whole=whole, 
                fs=fs, 
                log=log,
                mode="fft",
            )
            h *= rowh
    # instead, move all SOS onto batch dim, compute response, then move back
    else:  
        sos = sos.view(bs * n_sections, 6)
        w, sosh = freqz(
            sos[:, :3], 
            sos[:, 3:], 
            worN=worN, 
            whole=whole, 
            fs=fs, 
            log=log,
            mode="fft",
        )
        sosh = sosh.view(bs, n_sections, -1)
        for rowh in torch.chunk(sosh, n_sections, dim=1):
            rowh = rowh.view(bs, -1)
            h *= rowh
    return w, h
