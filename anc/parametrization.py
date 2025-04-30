import numpy as np
from dsp.filters import parametric_eq, sos_general_parametric


class Parametrization:
    def __init__(self):
        pass

    def decode(self, x):
        return x
    
    def bounds(self):
        return None
    
    def constraints(self):
        return []
    
    def __call__(self, x):
        return self.decode(x)


class ParametrizationEQ(Parametrization):
    def __init__(self, n_sections, ftypes, fs):
        assert len(ftypes) == n_sections
        super().__init__()
        self.n_sections = n_sections
        self.ftypes = ftypes
        self.fs = fs
        self.dim = 3 * n_sections
        
    def params_to_sos(self, f, Q, g):
        # PEQ to SOS
        sos = []
        for i in range(self.n_sections):
            bq = parametric_eq(f[i], Q[i], g[i], self.fs, self.ftypes[i]) 
            sos += [bq.reshape(1, -1)]
        sos = np.concatenate(sos, 0)
        return sos
    
    def decode(self, x):
        f, Q, g = np.array_split(x, 3)
        sos = self.params_to_sos(f, Q, g)
        return sos
    
    def bounds(self):
        f_min, f_max = 1., 20000.
        Q_min, Q_max = 0.1, 10
        g_min, g_max = -10, 10

        bounds = []
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        bounds += [(g_min, g_max)] * self.n_sections
        return bounds

        
class ParametrizationGeneral(Parametrization):
    def __init__(self, n_sections, fs):
        super().__init__()
        self.n_sections = n_sections
        self.fs = fs
        self.dim = 5 * n_sections
        
    def params_to_sos(self, f1, Q1, g1, f2, Q2):
        sos = []
        for i in range(self.n_sections):
            b0, b1, b2 = sos_general_parametric(f1[i], Q1[i], g1[i], self.fs, "vv")
            a0, a1, a2 = sos_general_parametric(f2[i], Q2[i], 0, self.fs, "vv")
            bq = np.asarray([b0, b1, b2, a0, a1, a2]) / a0
            assert bq[3] == 1.0
            sos += [bq.reshape(1, -1)]
        sos = np.concatenate(sos, 0)
        return sos
    
    def decode(self, x):
        f1, Q1, g1, f2, Q2 = np.array_split(x, 5)
        sos = self.params_to_sos(f1, Q1, g1, f2, Q2)
        return sos
        
    def bounds(self):
        f_min, f_max = 1., 20000.
        Q_min, Q_max = 0.1, 10
        g_min, g_max = -10, 10

        bounds = []
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        bounds += [(g_min, g_max)] * self.n_sections
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        return bounds
        