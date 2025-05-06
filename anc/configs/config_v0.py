import numpy as np
from scipy.stats import beta
from common.math import logscale


n_fft = 512 # 256, 4096, 8192, 16384
log = True

## weighting function
if log:
    freqs = logscale(0, 1, n_fft)
else:
    freqs = np.linspace(0, 1, n_fft)
    
a, b = 1.2, 20
weighting_feedback = beta(a, b).pdf(freqs) + 1e-6
weighting_feedback /= weighting_feedback.max()

a, b = 1, 10
weighting_feedforward = beta(a, b).pdf(freqs) + 1e-6
weighting_feedforward /= weighting_feedforward.max()

freqs *= 0.5

## constraints
gain_margin = 3 # 6
phase_margin = 30 # 45

gain_max = 6

freq_low = 10
freq_high = 2000
gain_open_max = 0
