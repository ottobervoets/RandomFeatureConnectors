import numpy as np
import scipy


class RFCNetwork:
    def __init__(self, N, M, aperture=8, seed=294369130659753536483103517623731383367):
        rng = np.random.default_rng(seed)

        self.N = N
        self.M = M
        self.aperture = aperture

        self.c = []
        self.p = []
        self.number_of_patterns_stored = 0
        self.pattern_length = []
        self.lr_c = 0.5

        # Create the matrix W

        self.W = np.array(rng.normal(0, 1, (N, N)))

        # Create random vectors W_in, r, and b
        self.W_in = rng.normal(0,1.2, self.N)
        self.W_out = np.array(rng.random(N))
        self.F = np.array(rng.normal(0, 1, (N, M)))

        # a**2 W F' F spectral radius of 1.4
        desired_radius = 1.4
        eigenvalues = np.linalg.eigvals(self.W @ self.F @ np.transpose(self.F))
        rho = np.max(np.abs(eigenvalues))
        a = np.power(desired_radius/rho, 1/3)
        self.W = a*self.W
        self.F = a*self.F


        self.r = np.array(rng.random(N))
        self.r_initial = self.r.copy()
        self.z = np.transpose(self.F) @ self.r_initial.copy()  # right??

        self.b = rng.normal(0, 0.2, N)
        self.D = rng.normal(0, 1, (self.N, self.M))  # random initialize D

        # Create a random matrix of size N x M

        # Create an empty list for c_list
        self.c_list = []


    def __repr__(self):
        return (f"RandomMatrixGenerator(N={self.N}, M={self.M}, "
                f"percent_non_zero={self.percent_non_zero})")

    def reset_r_z(self):
        self.r = self.r_initial.copy()
        self.z = np.transpose(self.F) @ self.r_initial.copy()  # right??

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        z = np.multiply(self.c[pattern_id], np.transpose(self.F) @ self.r)
        self.r = self.W @ self.F @ z + self.W_in @ self.D @ z + self.b

    def hallucinating(self, time, pattern_id=0, record_internal=False, record_y=False):
        recording_internal = []
        recording_y = []
        for t in range(time):
            self.one_step_hallucinating(pattern_id)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(self.W_out @ self.r)
        return recording_internal, recording_y

    def load_pattern_in_c(self, pattern, n_adapt, washout):

        self.c.append(np.ones(self.M))  # start at 1 conceptor
        self.pattern_length.append(len(pattern))
        # conceptor weight adaptation
        z = np.transpose(self.F) @ self.r_initial.copy()  # right??

        for t in range(n_adapt):
            z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ np.tanh((self.W @ self.F) @ z +
                                                                                        self.W_in * np.atleast_1d(pattern[t]) +
                                                                                        self.b)
            if t > washout:
                self.c[self.number_of_patterns_stored] = self.c[self.number_of_patterns_stored] + self.lr_c * (
                        z ** 2 - np.multiply(self.c[self.number_of_patterns_stored],z ** 2) -
                        self.aperture ** -2 * self.c[self.number_of_patterns_stored]
                )

    def record_r_z(self, n_harvest: int, pattern: list[float]):
        record_r = []
        record_z = []

        self.reset_r_z()
        # todo: add washout?
        for t in range(n_harvest):
            self.r = np.tanh(self.W @ self.F @ self.z + self.W_in * pattern[t] + self.b)
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ self.r

            record_r.append(self.r)
            record_z.append(self.z)

        return record_r, record_z

    def compute_D(self, z_recordings, patterns, beta_D):
        res = scipy.optimize.minimize(self.__D_objective, x0=self.D.flatten(), args=(z_recordings, patterns, beta_D))
        print(f"D objective = {self.__D_objective(self.D.flatten(), z_recordings, patterns, beta_D)}")
        return res.x.reshape(self.N, self.M)

    def __D_objective(self, D_flat, z_recordings, patterns, beta_D) -> float:

        D = D_flat.reshape(self.N, self.M)  # Reshape the flattened D into matrix form
        total_loss = 0
        for pattern, z_recording in zip(patterns, z_recordings):
            for pattern_step, z in zip(pattern, z_recording):
                total_loss += np.linalg.norm(self.W_in * pattern_step - D @ z, 2)
        # Add the regularization term
        total_loss += beta_D ** 2 * np.linalg.norm(D, 2)
        print(total_loss)
        return total_loss

    def compute_W_out(self, r_recordings, patterns, beta_W_out):
        res = scipy.optimize.minimize(self.__W_out_objective, x0=self.W_out.flatten(),
                                      args=(r_recordings, patterns, beta_W_out))
        print(f"W objective = {self.__W_out_objective(self.D.flatten(), r_recordings, patterns, beta_W_out)}")
        return res.x.reshape(self.N, self.M)

    def __W_out_objective(self, W_flat, r_recordings, patterns, beta_W_out) -> float:
        W = W_flat.reshape(self.N, self.M)  # Reshape the flattened D into matrix form
        total_loss = 0
        for pattern, r_recording in zip(patterns, r_recordings):
            for pattern_step, r in zip(pattern, r_recording):
                total_loss += np.linalg.norm(pattern_step - W @ np.atleast_1d(r), 2)
        # Add the regularization term
        total_loss += beta_W_out ** 2 * np.linalg.norm(W, 2)
        return total_loss

    def store_patterns(self, patterns, n_adapt, washout, n_harvest, beta_D, beta_W_out):
        z_recordings = []
        r_recordings = []

        for pattern in patterns:
            self.load_pattern_in_c(pattern, n_adapt, washout)
            record_r, record_z = self.record_r_z(pattern=pattern, n_harvest=n_harvest)

            z_recordings.append(record_z)
            r_recordings.append(record_r)

            self.number_of_patterns_stored += 1
            self.c_list.append(np.ones(self.M))  # extend c

        self.D = self.compute_D(z_recordings, patterns, beta_D)
        self.W_out = self.compute_W_out(r_recordings, patterns, beta_W_out)
