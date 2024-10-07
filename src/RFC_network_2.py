import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from src.RFC_network import RFCNetwork


class RFCNetwork2(RFCNetwork):

    def __init__(self, N, M, spectral_radius=1.4, lr_c=0.5, aperture=4, seed=294369130659753536483103517623731383366):
        super().__init__(N, M, spectral_radius=1.4, lr_c=0.5, aperture=4, seed=294369130659753536483103517623731383366)

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        # print(self.D @ self.z)
        self.r = np.tanh(self.G @ self.z + self.D @ self.z + self.b)
        if self.number_of_patterns_stored == 0:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r

    def compute_D_rigde(self, z_recordings, p_recordings, beta_D, option=1):
        Q = np.reshape(np.hstack(p_recordings), (1, len(p_recordings) * len(p_recordings[0])))

        y = np.transpose(np.outer(self.W_in, Q))  # Q = 1 x 1600
        # y = Q
        X = np.vstack(z_recordings)
        ridge = Ridge(alpha=beta_D, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        # return D_optimized

        # Q = np.reshape(np.hstack(p_recordings), (1, len(p_recordings) * len(p_recordings[0])))
        # Z = np.vstack(z_recordings).T
        # D_new = (np.linalg.inv(Z @ Z.T + beta_D * np.identity(self.M)) @ Z @ y.T).T
        # print(np.shape(D_optimized), np.shape(D_new))
        # print(np.linalg.norm(D_optimized - D_new.T, 2))
        return D_optimized
