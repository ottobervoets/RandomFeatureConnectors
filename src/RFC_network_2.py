import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from src.RFC_network import RFCNetwork


class RFCNetwork2(RFCNetwork):

    def __init__(self, N, M, signal_dim=1, spectral_radius=1.4, lr_c=0.5, aperture=4, seed=294369130659753536483103517623731383366):
        super().__init__(N, M, signal_dim=signal_dim, spectral_radius=spectral_radius,
                         lr_c=lr_c, aperture=aperture, seed=seed)

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        # print(self.D @ self.z)
        self.r = np.tanh(self.G @ self.z + self.D @ self.z + self.b)
        if self.number_of_patterns_stored == 0:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r

    def compute_D_rigde(self, z_recordings, p_recordings, beta_D, option=1):
        p_stacked = np.hstack(p_recordings)
        y = [self.W_in @ np.atleast_1d(p_t) for p_t in p_stacked]
        X = np.vstack(z_recordings)
        ridge = Ridge(alpha=beta_D, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        return D_optimized
