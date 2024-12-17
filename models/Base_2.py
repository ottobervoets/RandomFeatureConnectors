import numpy as np

from models.base_rfc import BaseRFC


class Base_2(BaseRFC):
    def construct_c(self, patterns, n_washout: int, n_harvest: int, matrix_conceptor: bool = True, **kwargs):
        for name, pattern in patterns.items():
            aperture = kwargs.get("aperture_" + name)
            print(name)
            self.r = np.zeros(self.N)
            for t in range(n_washout):
                self.drive_base_network(pattern[t])
            collected_z = np.zeros(shape=(n_harvest, self.M))
            if matrix_conceptor:
                collected_r = np.zeros(shape=(n_harvest, self.N))
            for t in range(n_washout, n_washout + n_harvest):
                self.drive_base_network(pattern[t])
                collected_z[t - n_washout] = np.transpose(self.F) @ self.r
                if matrix_conceptor:
                    collected_r[t - n_washout] = self.r
            if matrix_conceptor:
                self.matrix_conceptor[name] = self.compute_matrix_conceptor(collected_r, aperture)
                I = np.identity(self.N)
                self.memory = np.linalg.inv(
                    I + np.linalg.inv(
                        self.memory @ np.linalg.inv(I - self.memory) + self.matrix_conceptor[name] @
                        np.linalg.inv(I - self.matrix_conceptor[name])))
                _, S, _ = np.linalg.svd(self.memory)
                # plt.plot(S)
                # plt.title(f"Memory eigenvalue after adding {name}")
                # plt.show()

                if self.verbose:
                    print(f"Singular value sum = {sum(S)}, meaning a quota of {sum(S)/self.N}")

            mean_z_squared = np.mean(collected_z ** 2, axis=0)
            conceptor_weights = mean_z_squared / (mean_z_squared + float(aperture) ** -2)
            self.c[name] = conceptor_weights
        if self.verbose:
            print("conceptors constructed", np.shape(self.c))

    def drive_base_network(self, pattern):
        self.r = self.W_in @ np.atleast_1d(pattern) + self.W @ self.r

