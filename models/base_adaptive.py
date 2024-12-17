import numpy as np

from models.base_rfc import BaseRFC


class Base_adaptive(BaseRFC):
    def construct_c(self, patterns, n_washout: int, n_harvest: int, matrix_conceptor: bool = True, lr_c: float = 0.5, **kwargs):
        for name, pattern in patterns.items():
            aperture = kwargs.get("aperture_" + name)

            self.c[name] = np.ones(self.M)
            self.r = np.zeros(self.N)
            z = np.transpose(self.F) @ self.r  # right??
            print(self.G.shape)
            for t in range(n_washout + n_harvest):
                z = np.diag(self.c[name]) @ np.transpose(self.F) @ np.tanh(
                    self.G @ z +
                    self.W_in @ np.atleast_1d(
                        pattern[t]) +
                    self.b)
                if t > n_washout:
                    self.c[name] = self.c[name] + lr_c * (
                            z ** 2 - np.multiply(self.c[name], z ** 2) -
                            aperture ** -2 * self.c[name]
                    )
            self.last_training_z_state[name] = z
        if self.verbose:
            print("conceptors constructed", np.shape(self.c))


