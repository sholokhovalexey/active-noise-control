import numpy as np
import librosa


def delta_function(size, delay=0):
    assert delay < size
    ir = np.zeros(size)
    ir[delay] = 1.
    return ir


def resample(x, fs, fs_new, method="soxr_hq", axis=-1):
    scale = fs / fs_new
    y = scale * librosa.resample(
        x, 
        orig_sr=fs, 
        target_sr=fs_new, 
        res_type=method, 
        fix=True, 
        scale=False, 
        axis=axis,
    )
    return y


def delay(x, n_samples):
    assert n_samples >= 0
    length = x.shape[-1]
    return np.pad(x, (n_samples, 0))[..., :length]


def conv_causal(x, h):
    causal_padding = h.shape[-1] - 1
    x_pad = np.pad(x, (causal_padding, 0))
    y = np.convolve(x_pad, h, mode="valid")
    return y


def mag2db(x):
    return librosa.amplitude_to_db(x)


def db2mag(x):
    return librosa.db_to_amplitude(x)


def pow2db(x):
    return librosa.power_to_db(x)


def db2pow(x):
    return librosa.db_to_power(x)




