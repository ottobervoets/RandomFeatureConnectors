import numpy as np
from sklearn.linear_model import Ridge


class BaseRFC:
    def __init__(self,
                 N: int,
                 M: int,
                 signal_dim: int = 1,
                 aperture: float = 4,
                 spectral_radius: float = 1.4,
                 W_in_mean: float = 0,
                 W_in_std: float = 1.2,
                 b_mean: float = 0,
                 b_std: float = 0.2,
                 W_sparseness: float = 0.1,
                 W_sr: float = None,
                 reproducible: bool = False,
                 seed: int = 294369130659753536483103517623731383366,
                 verbose: bool = False,
                 **kwargs: object) -> None:

        if reproducible:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self.verbose = verbose
        if(self.verbose):
            print("default constructor")
        self.N = N
        self.M = M
        self.signal_dim = signal_dim
        self.aperture = aperture

        self.c = []
        self.c_matrix = []
        self.number_of_patterns_stored = 0
        self.last_training_z_state = []

        # Create Matrices
        if W_sr is None:
            W_sr = spectral_radius

        self.W = self.create_W(sparseness=W_sparseness, W_spectral_radius=W_sr)
        self.W_in = np.array(self.rng.normal(W_in_mean, W_in_std, (self.N, self.signal_dim)))
        self.W_out = np.array(self.rng.random((self.signal_dim, self.N)))
        self.b = self.rng.normal(b_mean, b_std, self.N)

        self.F = self.create_F(**kwargs)
        self.F_transpose = np.transpose(self.F)
        self.M_identity_F = np.identity(self.M) @ self.F_transpose
        self.G = self.create_G(**kwargs)
        self.spectral_radius_FG(spectral_radius) #make sure spectral radius is as desired

        self.z_initial = self.rng.normal(0, 0.5, self.M)
        self.z = self.z_initial.copy()
        self.r = self.G @ self.z_initial.copy()

        self.training_patterns = None


    def create_W(self, sparseness=0.1, W_spectral_radius=None):
        total_elements = self.N ** 2
        num_non_zero = int(total_elements * sparseness)

        W_flat = np.zeros(total_elements)
        non_zero_indices = self.rng.choice(total_elements, num_non_zero, replace=False)
        W_flat[non_zero_indices] = self.rng.normal(0, 1, num_non_zero)

        W_sparse = W_flat.reshape(self.N, self.N)
        eigenvalues = np.linalg.eigvals(W_sparse)
        rho = np.max(np.abs(eigenvalues))
        if self.verbose:
            print("spectral radius of W:", np.max(np.abs(np.linalg.eigvals(W_sparse * (W_spectral_radius / rho)))))
        if W_spectral_radius is None:
            return W_sparse
        else:
            return W_sparse * (W_spectral_radius / rho)

    def create_F(self, **kwargs):
        if self.verbose:
            print("created F from random normal")
        return np.array(self.rng.normal(0, 1, (self.N, self.M)))

    def create_G(self, **kwargs):
        if self.verbose:
            print("created G =  WF")
        return self.W @ self.F

    def spectral_radius_FG(self, spectral_radius):
        eigenvalues = np.linalg.eigvals(np.transpose(self.F) @ self.G)
        rho = np.max(np.abs(eigenvalues))
        a = np.power(spectral_radius / rho, 1 / 2)
        self.F = a * self.F
        self.G = a * self.G
        if self.verbose:
            print(f"spectral radius of F' G = {np.max(np.abs(np.linalg.eigvals(np.transpose(self.F) @ self.G)))} ")

    def set_z_state(self, pattern_id: int = None):  # may add noise later on
        if pattern_id is not None:
            self.z = self.last_training_z_state[pattern_id]
            return
        self.z = np.zeros(self.M)

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        self.r = np.tanh(self.G @ self.z + self.D @ self.z + self.b)

        if self.number_of_patterns_stored == 0:
            self.z = self.M_identity_F @ self.r
        else:
            self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r

    def one_step_driving(self, pattern, pattern_id: int = None):
        if self.signal_dim == 1:
            pattern = np.atleast_1d(pattern)
        self.r = np.tanh(self.G @ self.z + self.W_in @ pattern + self.b)
        if pattern_id is None:
            # self.z = self.M_identity_F @ self.r
            self.z = np.identity(self.M) @ self.F_transpose @ self.r
        else:
            self.z = self.c_matrix[pattern_id] @ self.F_transpose @ self.r


    def hallucinating(self, length, pattern_id: int = None, record_internal: bool = False, record_y: bool = False):
        self.set_z_state(pattern_id)

        recording_internal = []
        recording_y = []

        for t in range(length):
            self.one_step_hallucinating(pattern_id)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(self.W_out @ self.r)
        return recording_internal, recording_y

    # def one_step_drive_simple(self, pattern):
    #     if self.signal_dim < 2:
    #         pattern = np.atleast_1d(pattern)
    #     self.r = np.tanh(self.W @ self.r + self.W_in @ pattern + self.b)
    #
    # def construct_c(self, patterns, n_washout: int, n_harvest: int):
    #     for pattern in patterns:
    #         self.r = np.zeros(self.N)
    #         for t in range(n_washout):
    #             self.one_step_drive_simple(pattern[t])
    #         collected_z = np.zeros(shape=(n_harvest, self.M))
    #         for t in range(n_washout, n_washout + n_harvest):
    #             self.one_step_drive_simple(pattern[t])
    #             collected_z[t-n_washout] = self.F_transpose @ self.r
    #
    #         mean_z_squared = np.mean(collected_z ** 2, axis=0)
    #         conceptor_weights = mean_z_squared / (mean_z_squared + self.aperture ** -2)
    #         self.c.append(conceptor_weights)
    #         self.c_matrix.append(np.diag(conceptor_weights))
    #         self.last_training_z_state.append(self.z)
    #     if self.verbose:
    #         print("conceptors constructed", np.shape(self.c))

    def construct_c(self, patterns, n_washout: int, n_harvest: int):
        for pattern in patterns:
            for t in range(n_washout):
                self.one_step_driving(pattern[t], pattern_id=None)
            collected_z = np.zeros(shape=(n_harvest, self.M))
            for t in range(n_washout, n_washout + n_harvest):
                self.one_step_driving(pattern[t], pattern_id=None)
                collected_z[t-n_washout] = self.z

            mean_z_squared = np.mean(collected_z ** 2, axis=0)
            conceptor_weights = mean_z_squared / (mean_z_squared + self.aperture ** -2)
            self.c.append(conceptor_weights)
            self.c_matrix.append(np.diag(conceptor_weights))
            self.last_training_z_state.append(self.z)
        if self.verbose:
            print("conceptors constructed", np.shape(self.c))

    def record_r_z(self, pattern_id, n_washout: int, n_harvest: int, pattern):
        record_r = np.zeros((n_harvest,self.N))
        record_z = np.zeros((n_harvest,self.M))
        record_p = np.zeros((n_harvest, self.signal_dim))

        self.set_z_state(None)
        for t in range(n_washout):
            self.one_step_driving(pattern[t], pattern_id=pattern_id)

        for t in range(n_washout, n_washout + n_harvest):
            record_z[t-n_washout] = self.z
            self.one_step_driving(pattern[t], pattern_id=pattern_id)
            record_r[t-n_washout] = self.r
            record_p[[t-n_washout]]= pattern[t]

        return record_r, record_z, record_p

    def compute_W_out_ridge(self, r_recordings, patterns, beta_W_out):
        y = np.array(np.vstack(patterns))
        X = np.vstack(r_recordings)
        ridge = Ridge(alpha=beta_W_out, fit_intercept=False)
        ridge.fit(X, y)

        W_out_optimized = ridge.coef_
        return W_out_optimized

    def compute_D_rigde(self, z_recordings, p_recordings, beta_D):
        p_stacked = np.vstack(p_recordings)
        y = [self.W_in @ np.atleast_1d(p_t) for p_t in p_stacked]
        X = np.vstack(z_recordings)
        ridge = Ridge(alpha=beta_D, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        if self.verbose:
            print("D_optimized", np.shape(D_optimized))
        return D_optimized

    def print_NRMSEs(self, z_recordings, r_recordings, p_recordings):
        D = 0
        W_out = 0
        for z_recording, r_recording, p_recording in zip(z_recordings, r_recordings, p_recordings):
            for z_t, r_t, p_t in zip(z_recording, r_recording, p_recording):
                D += np.linalg.norm(self.W_in @ np.atleast_1d(p_t) - self.D @ z_t, 2)
                W_out += np.mean((p_t - self.W_out @ r_t) ** 2)
        num_z = len(z_recordings[0][0]) * len(z_recordings[0]) * len(z_recordings)
        num_p = len(p_recordings[0]) * len(p_recordings)
        if self.verbose:
            print(f"D RMSE {np.sqrt(D)}, W_out {np.sqrt(W_out)}, G = {np.linalg.norm(self.G, 2)}")
            print(f"D NRMSE {np.sqrt(D / num_z)}, W_out {np.sqrt(W_out / num_p)}, G = {np.linalg.norm(self.G, 2)}")

    def compute_G(self, z_recordings, beta_G):
        y = [self.G @ np.transpose(z_recoding) for z_recoding in z_recordings]
        y = np.transpose(np.hstack(y))
        X = np.vstack(z_recordings)
        ridge = Ridge(alpha=beta_G, fit_intercept=False)
        ridge.fit(X, y)
        G_optimized = ridge.coef_
        if self.verbose:
            print(f"Mean absolute size of G {np.linalg.norm
            (G_optimized):.2f} which was {np.linalg.norm(self.G):.2f}, so a decrease"
                  f" of {(np.linalg.norm(self.G) - np.linalg.norm(G_optimized)) / np.linalg.norm(self.G):.2f}")
        return G_optimized

    def store_patterns(self, training_patterns, washout, n_harvest, beta_D, beta_W_out, beta_G,
                       noise_std=None, **kwargs):
        self.training_patterns = training_patterns
        if noise_std is not None:
            for i in range(len(training_patterns)):
                noise = self.rng.normal(0, noise_std, (len(training_patterns[i]), self.signal_dim))
                training_patterns[i] = training_patterns[i] + noise
        z_recordings = []
        r_recordings = []
        p_recordings = []

        self.construct_c(training_patterns, washout, n_harvest)

        for pattern_id in range(len(training_patterns)):
            record_r, record_z, record_p = self.record_r_z(pattern=training_patterns[pattern_id],
                                                           n_harvest=n_harvest,
                                                           n_washout=washout,
                                                           pattern_id=pattern_id)
            z_recordings.append(record_z)
            r_recordings.append(record_r)
            p_recordings.append(record_p)

            self.number_of_patterns_stored += 1

        self.D = self.compute_D_rigde(z_recordings, p_recordings, beta_D)
        self.W_out = self.compute_W_out_ridge(r_recordings, p_recordings, beta_W_out)
        self.G = self.compute_G(z_recordings, beta_G)
        self.print_NRMSEs(z_recordings, r_recordings, p_recordings)

    def record_chaotic(self, length, pattern_id):
        _, y_recording = self.hallucinating(length=length, pattern_id=pattern_id, record_internal=False, record_y=True)
        return y_recording
