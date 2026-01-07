import numpy as np
import yaml
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

    def load(self, fname):
        with open(fname, "r") as f:
            data = yaml.load(f, Loader=yaml.loader.SafeLoader)

        if not data["name"] == self.__class__.__name__:
            print(f'Wrong parametrization: {data["name"]} != {self.__class__.__name__}')
            
        x = np.array(data["params"], dtype=np.float64)
        return x


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

    def encode(self, x):
        x = np.array(x)
        assert x.shape[1] == 3
        f, Q, g = np.array_split(x, 3, axis=1)
        return np.r_[f, Q, g].ravel()
        
    def bounds(self):
        f_min, f_max = 1., 20000.
        Q_min, Q_max = 0.1, 10
        g_min, g_max = -20, 20

        bounds = []
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        bounds += [(g_min, g_max)] * self.n_sections
        return bounds

    def save(self, fname, x):
        data_dict = {}
        data_dict["name"] = self.__class__.__name__
        data_dict["fs"] = self.fs
        data_dict["ftypes"] = self.ftypes
        data_dict["params"] = [str(x[i]) for i in range(len(x))]
        with open(fname, "w") as file:
            yaml.dump(data_dict, file, default_flow_style=False)

    
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
        g_min, g_max = -20, 20

        bounds = []
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        bounds += [(g_min, g_max)] * self.n_sections
        bounds += [(f_min, f_max)] * self.n_sections
        bounds += [(Q_min, Q_max )] * self.n_sections
        return bounds

    def save(self, fname, x):
        data_dict = {}
        data_dict["name"] = self.__class__.__name__
        data_dict["fs"] = self.fs
        data_dict["ftypes"] = self.ftypes
        data_dict["params"] = [str(x[i]) for i in range(len(x))]
        with open(fname, "w") as file:
            yaml.dump(data_dict, file, default_flow_style=False)