import numpy as np
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

class RFCNetwork:
    def __init__(self, N, M,spectral_radius = 1.4, lr_c=0.5, aperture=4, seed=294369130659753536483103517623731383366,
                 F_method="random", **kwargs):
        self.rng = np.random.default_rng(seed)

        self.N = N
        self.M = M
        self.aperture = aperture

        self.c = []
        self.number_of_patterns_stored = 0
        self.lr_c = lr_c

        # Create the matrix W
        self.W = np.array(self.rng.normal(0, 1, (N, N)))

        # Create random vectors W_in, r, and b
        self.W_in = np.array(self.rng.normal(0, 1.2, self.N))
        self.W_out = np.array(self.rng.random(N))


        self.b = self.rng.normal(0, 0.2, N)
        self.D = self.rng.normal(0, 1, (self.N, self.M))  # random initialize D


        # Create an empty list for conceptors
        self.c_list = []
        match F_method:
            case "random":

                self.F = np.array(self.rng.normal(0, 1, (N, M)))
            case "white_noise":
                self.construct_F_white_noise()
            case "patterns":
                if 'patterns' in kwargs:
                    self.construct_F_patterns(patterns=kwargs['patterns'])
                else:
                    raise ValueError("To construct F based on patterns, a list of patterns is needed.")
            case _:
                raise ValueError("Construction method not supported. Allowed methods: \"random\", \"white_noise\", \"patterns\"")

        self.G = self.W @ self.F

        eigenvalues = np.linalg.eigvals(np.transpose(self.F) @ self.G)
        rho = np.max(np.abs(eigenvalues))
        a = np.power(spectral_radius / rho, 1 / 2)
        self.F = a * self.F
        self.G = a * self.G

        self.z_initial = self.rng.normal(0, 0.5, self.M)
        self.z = self.z_initial.copy()
        self.r = self.G @ self.z_initial.copy()

        print(f"spectral radius of F' G = {np.max(np.abs(np.linalg.eigvals(np.transpose(self.F) @ self.G)))} ")

    def construct_F_white_noise(self, sample_rate = 10, washout = 200):
        white_noise_sequence = self.rng.uniform(-1,1, self.M * sample_rate + washout)
        map_F = []
        for idx in range(len(white_noise_sequence)):
            self.r = self.G_tilde @ self.r + self.W_in * white_noise_sequence[idx]
            if idx > washout and idx % sample_rate == 0:
                map_F.append(self.r)

        self.F = np.array(map_F)
        print("map F:", np.shape(self.F))


    def construct_F_patterns(self, patterns, washout = 200): #todo: sample in advance
        tot_len = sum(len(pattern) for pattern in patterns)
        frequency = tot_len - len(patterns) * washout
        map_F = []
        for pattern in patterns:
            self.r = self.rng.normal(0, 0.5, self.N)
            for idx in range(len(pattern)):
                self.r = self.G_tilde @ self.r + self.W_in * p[idx]
                if idx > washout and idx % sample_rate == 0:
                    map_F.append(self.r)

        self.F = np.array(map_F)

    def __repr__(self):
        return f"RandomMatrixGenerator(N={self.N}, M={self.M}"

    def reset_r_z(self):
        self.z = self.z_initial  # right??
        self.r = self.G @ self.z

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        # print(self.D @ self.z)
        self.r = np.tanh(self.G @ self.z + self.W_in * float(self.D @ self.z) + self.b)
        if self.number_of_patterns_stored == 0:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r


    def hallucinating(self, time, pattern_id=0, record_internal=False, record_y=False):
        self.reset_r_z()
        recording_internal = []
        recording_y = []
        for t in range(time):
            self.one_step_hallucinating(pattern_id)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(float(self.W_out @ self.r))
        return recording_internal, recording_y

    def load_pattern_in_c(self, pattern, n_adapt, washout):

        self.c.append(np.ones(self.M))  # start at 1 conceptor
        # conceptor weight adaptation
        self.reset_r_z()

        for t in range(n_adapt):
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ np.tanh(self.G @ self.z +
                                                                                                 self.W_in *
                                                                                                 np.atleast_1d(
                                                                                                     pattern[t]) +
                                                                                                 self.b)
            if t > washout:
                self.c[self.number_of_patterns_stored] = self.c[self.number_of_patterns_stored] + self.lr_c * (
                        self.z ** 2 - np.multiply(self.c[self.number_of_patterns_stored], self.z ** 2) -
                        self.aperture ** -2 * self.c[self.number_of_patterns_stored]
                )

    def record_r_z(self, n_washout: int, n_harvest: int, pattern: list[float]):
        record_r = []
        record_z = []
        record_p = []

        self.reset_r_z()
        for t in range(n_washout):
            self.r = np.tanh(self.G @ self.z + self.W_in * pattern[t] + self.b)
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ self.r

        for t in range(n_harvest):
            record_z.append(self.z)
            self.r = np.tanh(self.G @ self.z + self.W_in * pattern[t] + self.b)
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ self.r
            record_r.append(self.r)
            record_p.append(pattern[t])

        return record_r, record_z, record_p

    def compute_W_out_ridge(self, r_recordings, patterns, beta_W_out):
        y = np.array(np.hstack(patterns))
        X = np.vstack(r_recordings)
        ridge = Ridge(alpha=beta_W_out, fit_intercept=False)
        ridge.fit(X, y)

        W_out_optimized = ridge.coef_.T

        return W_out_optimized

    def compute_D_rigde(self, z_recordings, p_recordings, beta_D, option=1):
        Q = np.reshape(np.hstack(p_recordings), (1, len(p_recordings) * len(p_recordings[0])))

        y = np.transpose(np.outer(self.W_in, Q))  # Q = 1 x 1600
        # y = Q
        X = np.vstack(z_recordings)
        ridge = Ridge(alpha=beta_D, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        # return D_optimized

        Q = np.reshape(np.hstack(p_recordings), (1, len(p_recordings) * len(p_recordings[0])))
        Z = np.vstack(z_recordings).T
        D_new = (np.linalg.inv(Z @ Z.T + beta_D * np.identity(self.M)) @ Z @ Q.T).T
        print(np.shape(D_optimized), np.shape(D_new))
        # print(np.linalg.norm(D_optimized - D_new.T, 2))
        return D_new

    def print_NRMSEs(self, z_recordings, r_recordings, p_recordings, beta_G):
        D = 0
        W_out = 0
        # print(np.shape(self.D), np.shape(self.z))
        print(f"Eigen values of D{np.mean(np.linalg.eigvals(self.D @ self.D.T))}")
        for z_recording, r_recording, p_recording in zip(z_recordings, r_recordings, p_recordings):
            for z_t, r_t, p_t in zip(z_recording, r_recording, p_recording):
                D += np.linalg.norm(self.W_in * p_t - self.D @ z_t, 2)
                W_out += (p_t - self.W_out @ r_t) ** 2
        num_z = len(z_recordings[0][0])*len(z_recordings[0])*len(z_recordings)
        num_p = len(p_recordings[0]) * len(p_recordings)
        print(f"D RMSE {np.sqrt(D)}, W_out {np.sqrt(W_out)}, G = {beta_G * np.linalg.norm(self.G)}")
        print(f"D NRMSE {np.sqrt(D/num_z)}, W_out {np.sqrt(W_out/num_p)}, G = {beta_G * np.linalg.norm(self.G)}")

    def compute_G(self, z_recordings, beta_G):
        print(np.shape(z_recordings))
        y = [self.G @ np.transpose(z_recoding) for z_recoding in z_recordings]
        print(np.shape(y))
        y = np.transpose(np.hstack(y))
        print(np.shape(y))
        X = np.vstack(z_recordings)
        Z = np.transpose(X)
        G_alternative = (np.linalg.inv(Z @ Z.T + beta_G * np.identity(self.M)) @ Z @ (self.G @ Z).T).T
        # G_alternative = np.linalg.inv(Z @ Z.T + beta_G * np.identity(self.M))
        ridge = Ridge(alpha=beta_G, fit_intercept=False)
        ridge.fit(X, y)
        G_optimized = ridge.coef_
        print("difference", np.linalg.norm(G_alternative - G_optimized, 2))


        return G_optimized

    def store_patterns(self, patterns, n_adapt, washout, n_harvest, beta_D, beta_W_out, beta_G):
        z_recordings = []
        r_recordings = []
        p_recordings = []

        for pattern in patterns:
            self.load_pattern_in_c(pattern, n_adapt, washout)
            record_r, record_z, record_p = self.record_r_z(pattern=pattern, n_harvest=n_harvest, n_washout=washout)

            z_recordings.append(record_z)
            r_recordings.append(record_r)
            p_recordings.append(record_p)

            self.number_of_patterns_stored += 1
            self.c_list.append(np.ones(self.M))  # extend c


        # self.print_NRMSEs(z_recordings, r_recordings, p_recordings, beta_G)

        self.D = self.compute_D_rigde(z_recordings, p_recordings, beta_D)
        self.W_out = self.compute_W_out_ridge(r_recordings, p_recordings, beta_W_out)
        self.G = self.compute_G(z_recordings, beta_G)
        self.print_NRMSEs(z_recordings, r_recordings, p_recordings, beta_G)



# so Z_recordings = [recording_1,... where recording_1 = [r_0 for pattern 1. so
