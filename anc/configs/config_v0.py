from scipy.stats import beta
from common.math import logscale


n_fft = 256 # 256, 4096, 8192, 16384
log = True

# weighting function
if log:
    x = logscale(0, 1, n_fft)
else:
    x = np.linspace(0, 1, n_fft)
    
a, b = 1, 10
weighting = beta(a, b).pdf(x)

# constraints
gain_margin = 3 # 6
phase_margin = 30 # 45

gain_max = 6

freq_low = 10
freq_high = 2000
gain_open_max = 0
