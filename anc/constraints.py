import numpy as np
from dsp import mag2db, db2mag


#TODO: rename all
class FreqConstraint: 
    def __init__(self):
        pass
    
    def _unpack_args(self, *args): #TODO: keep only this func, no need for a class ??
        f = None
        if len(args) == 1:
            H = args[0]
        else:
            f, H = args[:2]
        return f, H
    
    def __call__(self):
        pass
    
    
class ConstraintLowFreq(FreqConstraint):
    def __init__(self, freq, gain_open_max):
        super().__init__()
        self.freq = freq
        self.gain_open_max = gain_open_max
        
    def __call__(self, *args):
        f, H = self._unpack_args(*args)
        mask = f < self.freq
        H_mag = mag2db(np.abs(H))
        return H_mag[mask] - self.gain_open_max
    
    
class ConstraintHighFreq(FreqConstraint):
    def __init__(self, freq, gain_open_max):
        super().__init__()
        self.freq = freq
        self.gain_open_max = gain_open_max
        
    def __call__(self, *args):
        f, H = self._unpack_args(*args)
        mask = f > self.freq
        H_mag = mag2db(np.abs(H))
        return H_mag[mask] - self.gain_open_max
    
#TODO: rename to ConstraintStability ?? Constraint*
class StabilityConstraint(FreqConstraint):
    def __init__(self, gain_margin=None, phase_margin=None):
        super().__init__()
        assert gain_margin is None or gain_margin > 0
        assert phase_margin is None or phase_margin > 0
        self.gain_margin = gain_margin
        self.phase_margin = phase_margin
        
    def __call__(self):
        pass
    
    
class ConstraintCircle(FreqConstraint):
    def __init__(self, gain_max):
        self.b = 1 / db2mag(gain_max)
        GM = 1 / (1 - self.b)
        self.gain_margin = mag2db(gain_max)
        PM = np.arccos(1 - self.b**2 / 2)
        self.phase_margin = np.rad2deg(PM)

    def __call__(self, *args):
        f, H = self._unpack_args(*args)
        return self.b - np.abs(1 + H)
    
    
class StabilityConstraintParabola(StabilityConstraint):
    def __init__(self, gain_margin, phase_margin):
        super().__init__(gain_margin, phase_margin)
        GM = db2mag(self.gain_margin)
        PM = np.deg2rad(self.phase_margin)
        self.b = - 1 / GM
        self.a = (self.b + np.cos(PM)) / np.sin(PM)**2

    def __call__(self, *args):
        f, H = self._unpack_args(*args)
        return self.b - H.real - self.a * H.imag**2


class StabilityConstraintHyperbola(StabilityConstraint):
    def __init__(self, gain_margin):
        super().__init__(gain_margin)
        GM = db2mag(self.gain_margin)
        self.a = 1 / GM
        PM = np.arccos(self.a * np.sqrt(2 - self.a**2))
        self.phase_margin = np.rad2deg(PM)
        
    def __call__(self, *args):
        f, H = self._unpack_args(*args)
        return np.abs(1 - H) - np.abs(1 + H) - 2 * self.a