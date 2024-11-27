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

        self.c = {}
        self.number_of_patterns_stored = 0
        self.last_training_z_state = {}

        # Create Matrices
        if W_sr is None:
            W_sr = spectral_radius

        self.W = self.create_W(sparseness=W_sparseness, W_spectral_radius=W_sr)
        self.W_in = np.array(self.rng.normal(W_in_mean, W_in_std, (self.N, self.signal_dim)))
        self.W_out = np.array(self.rng.random((self.signal_dim, self.N)))
        self.b = self.rng.normal(b_mean, b_std, self.N)

        self.F = self.create_F(**kwargs)
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

    def set_z_state(self, pattern_name: str = None):  # may add noise later on
        if pattern_name is not None:
            self.z = self.last_training_z_state[pattern_name]
            return
        self.z = np.zeros(self.M)

    def one_step_hallucinating(self, pattern_name=None):  # taken from page 113
        self.r = np.tanh(self.G @ self.z + self.D @ self.z + self.b)

        if self.number_of_patterns_stored == 0 or pattern_name is None:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_name]) @ np.transpose(self.F) @ self.r

    def one_step_driving(self, pattern, pattern_name: str = None):
        self.r = np.tanh(self.G @ self.z + self.W_in @ np.atleast_1d(pattern) + self.b)
        if pattern_name is None:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_name]) @ np.transpose(self.F) @ self.r

    def hallucinating(self, length, pattern_name: str = None, record_internal: bool = False, record_y: bool = False):
        self.set_z_state(pattern_name)

        recording_internal = []
        recording_y = []

        for t in range(length):
            self.one_step_hallucinating(pattern_name)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(self.W_out @ self.r)
        return recording_internal, recording_y

    def construct_c(self, patterns, n_washout: int, n_harvest: int, **kwargs):
        for name, pattern in patterns.items():
            aperture = kwargs.get("aperture_"+name)
            for t in range(n_washout):
                self.one_step_driving(pattern[t], pattern_name=None)
            collected_z = []
            for t in range(n_washout, n_washout + n_harvest):
                self.one_step_driving(pattern[t], pattern_name=None)
                collected_z.append(self.z)
            collected_z = np.array(collected_z)

            mean_z_squared = np.mean(collected_z ** 2, axis=0)
            conceptor_weights = mean_z_squared / (mean_z_squared + aperture ** -2)
            self.c[name] = conceptor_weights
            self.last_training_z_state[name] = self.z.copy()
        if self.verbose:
            print("conceptors constructed", np.shape(self.c))

    def record_r_z(self, pattern_name:str, n_washout: int, n_harvest: int, pattern):
        record_r = []
        record_z = []
        record_p = []

        self.set_z_state(None)
        for t in range(n_washout):
            self.one_step_driving(pattern[t], pattern_name=pattern_name)

        for t in range(n_washout, n_washout + n_harvest):
            record_z.append(self.z)
            self.one_step_driving(pattern[t], pattern_name=pattern_name)
            record_r.append(self.r)
            record_p.append(np.atleast_1d(pattern[t]))
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
            for name, training_pattern in training_patterns.items():
                noise = self.rng.normal(0, noise_std, (len(training_pattern), self.signal_dim))
                training_patterns[name] = training_patterns[name] + noise
        z_recordings = []
        r_recordings = []
        p_recordings = []

        self.construct_c(training_patterns, washout, n_harvest, **kwargs)

        for name, training_pattern in training_patterns.items():
            record_r, record_z, record_p = self.record_r_z(pattern=training_pattern,
                                                           n_harvest=n_harvest,
                                                           n_washout=washout,
                                                           pattern_name=name)
            z_recordings.append(record_z)
            r_recordings.append(record_r)
            p_recordings.append(record_p)

            self.number_of_patterns_stored += 1

        self.D = self.compute_D_rigde(z_recordings, p_recordings, beta_D)
        self.W_out = self.compute_W_out_ridge(r_recordings, p_recordings, beta_W_out)
        self.G = self.compute_G(z_recordings, beta_G)
        self.print_NRMSEs(z_recordings, r_recordings, p_recordings)

    def record_chaotic(self, length, pattern_name):
        _, y_recording = self.hallucinating(length=length, pattern_name=pattern_name, record_internal=False, record_y=True)
        return y_recording