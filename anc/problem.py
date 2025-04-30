import math
import numpy as np
import scipy.signal

from anc.constraints import (
    ConstraintCircle, 
    StabilityConstraintParabola, 
    ConstraintLowFreq,
    ConstraintHighFreq,
)

from common.general import compose
from common.cache import memoize
from dsp.filters import freqz, sosfreqz

from optim import Problem


class ProblemFreq(Problem):
    def __init__(self, fs, n_fft=512, logspace=False):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.initialized = False
        self.logspace = logspace
        # sosfreqz_fn = lambda sos: scipy.signal.sosfreqz(sos, worN=self.n_fft, fs=self.fs)
        sosfreqz_fn = lambda sos: sosfreqz(sos, worN=self.n_fft, fs=self.fs, log=self.logspace)
        self.sosfreqz = memoize(sosfreqz_fn)
        
    def init(self):
        self.initialized = True
        
    
class ProblemFeedback(ProblemFreq):
    def __init__(self, fs, n_fft=512, logspace=False):
        super().__init__(fs, n_fft, logspace)
        
        # sosfreqz_fn = lambda sos: scipy.signal.sosfreqz(sos, worN=self.n_fft, fs=self.fs)
        sosfreqz_fn = lambda sos: sosfreqz(sos, worN=self.n_fft, fs=self.fs, log=self.logspace)
        self.sosfreqz = memoize(sosfreqz_fn)
        def _open_loop(sos):
            freq, FB = self.sosfreqz(sos)
            return freq, self.SP * FB 
        self.tf_open_loop = memoize(lambda sos: _open_loop(sos))
        
    def open_loop(self, sos, n_fft=None):
        assert (sos[:, 3] == 1).all()
        # sos = self.params_to_sos(f, Q, g)
        return self.tf_open_loop(sos)
        
    def init(self, SP, weighting=None):
        # freq, self.SP = scipy.signal.freqz(SP, 1, worN=self.n_fft, fs=self.fs)
        freq, self.SP = freqz(SP, 1, worN=self.n_fft, fs=self.fs, log=self.logspace)
        if weighting is None:
            weighting = 1 #make_weighting_function(self.n_fft)
        self.weighting = weighting
        super().init()
    
    def sensitivity(self, sos):
        # sos = self.params_to_sos(f, Q, g)
        freq, FB = self.sosfreqz(sos)
        H = self.SP * FB # open loop
        S_fb = 1 / (1 + H)
        return freq, S_fb
        
    def objective(self, sos):
        assert self.initialized
        freq, S_fb = self.sensitivity(sos)
        loss = np.mean(self.weighting * (np.abs(S_fb)**2))
        # loss = np.mean(self.weighting * librosa.amplitude_to_db(np.abs(S_fb)))
        return loss
    
    def add_constraints(self, cfg):
        assert hasattr(self, "constraints")

        constraint_circle = ConstraintCircle(cfg.gain_max)
        def constraint_circle_fn(sos):
            w, H = self.open_loop(sos)
            return constraint_circle(H)
        # self.constraints["circle"] = constraint_circle_fn
        f = compose(np.max, constraint_circle)
        self.add_constraint(f, name="circle", keep_feasible=True)
            
        constraint_parabola = StabilityConstraintParabola(cfg.gain_margin, cfg.phase_margin)
        def constraint_parabola_fn(sos):
            w, H = self.open_loop(sos)
            return constraint_parabola(H)
        # self.constraints["parabola"] = constraint_parabola_fn
        f = compose(np.max, constraint_parabola_fn)
        self.add_constraint(f, name="parabola", keep_feasible=True)
        
        constraint_lowfreq = ConstraintLowFreq(cfg.freq_low, cfg.gain_open_max)
        def constraint_lowfreq_fn(sos):
            w, H = self.open_loop(sos)
            return constraint_lowfreq(w, H)
        # self.constraints["lowfreq"] = constraint_lowfreq_fn
        f = compose(np.max, constraint_lowfreq_fn)
        self.add_constraint(f, name="lowfreq", keep_feasible=True)
        
        constraint_highfreq = ConstraintHighFreq(cfg.freq_high, cfg.gain_open_max)
        def constraint_highfreq_fn(sos):
            w, H = self.open_loop(sos)
            return constraint_highfreq(w, H)
        # self.constraints["highfreq"] = constraint_highfreq_fn
        f = compose(np.max, constraint_highfreq_fn)
        self.add_constraint(f, name="highfreq", keep_feasible=True)
        
        
class ProblemFeedforward(ProblemFreq):
    def __init__(self, fs, n_fft=512, logspace=False):
        super().__init__(fs, n_fft, logspace)
        
    def init(self, PP, SP, FB=None, weighting=None):
        freq, self.PP = freqz(PP, 1, worN=self.n_fft, fs=self.fs, log=self.logspace)
        freq, self.SP = freqz(SP, 1, worN=self.n_fft, fs=self.fs, log=self.logspace)
        self.FB = 0 if FB is None else FB
        if weighting is None:
            weighting = 1 #make_weighting_function(self.n_fft)
        self.weighting = weighting
        super().init()
        
    def sensitivity(self, sos):
        freq, FF = self.sosfreqz(sos)
        S_ff = (self.PP - FF * self.SP) / self.PP
        return freq, S_ff
    
    def objective(self, sos):
        S_ff = self.sensitivity(sos)
        if isinstance(self.FB, np.ndarray):
            H = self.SP * self.FB # open loop
            S_ff /= (1 + H)
        loss = np.mean(self.weighting * (np.abs(S_ff)**2))
        # loss = np.mean(self.weighting * librosa.amplitude_to_db(np.abs(S_ff)))
        return loss