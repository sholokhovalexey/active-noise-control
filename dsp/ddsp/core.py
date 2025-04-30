import math
import numpy as np
import torch
import torch.nn.functional as F

from common.math import sin, cos, sqrt


def delay(x, n_samples):
    assert n_samples >= 0
    length = x.shape[-1]
    if isinstance(x, torch.Tensor):
        return F.pad(x, (n_samples, 0))[..., :length]
    else:
        return np.pad(x, (n_samples, 0))[..., :length]


def conv_causal(x, h):
    causal_padding = len(h) - 1
    if isinstance(x, torch.Tensor):
        x_pad = F.pad(x, (causal_padding, 0))
        h_inv = torch.flip(h, -1)
        y = F.conv1d(x_pad, h_inv, mode="valid")
    else:
        x_pad = np.pad(x, (causal_padding, 0))
        y = np.convolve(x_pad, h, mode="valid")
    return y


def to_wav_torch(x, is_batch=True):
    if x.ndim == 1:
        return torch.tensor(x).view(1, 1, -1)
    elif x.ndim == 2:
        N, T = x.shape
        if is_batch:
            return torch.tensor(x).unsqueeze(1)
        else:
            return torch.tensor(x).unsqueeze(0)
    else:
        return torch.tensor(x)


def to_wav_numpy(x, is_batch=True):
    x = x.detach().cpu()
    x = x.squeeze()
    return x.numpy()
    # if x.ndim == 1:
    #     return x
    # elif x.ndim == 2:
    #     N, T = x.shape
    #     return x
    # elif x.ndim == 3:
    #     B, C, T = x.shape

    #     return x
    # else:
    #     return torch.tensor(x)
