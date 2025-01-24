import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from models.matrix_conceptor_rebuild import nrmse


class MatrixConceptor:
    def __init__(self,
                 N: int,
                 signal_dim: int = 2,
                 aperture: float = 4,
                 spectral_radius: float = 1.4,
                 W_in_std: float = 1.2,
                 b_std: float = 0.4,
                 W_sparseness: float = 0.1,
                 W_sr: float = None,
                 reproducible: bool = False,
                 seed: int = 294369130659753536483103517623731383366,
                 verbose: bool = False,
                 plot_conceptors: bool = True,
                 **kwargs: object) -> None:
        if reproducible:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self.verbose = verbose
        self.bool_plt_c = plot_conceptors
        if (self.verbose):
            print("default constructor")
        self.N = N
        self.signal_dim = signal_dim
        self.aperture = aperture

        self.c = {}
        self.matrix_conceptor = {}
        self.memory = np.zeros(shape=(self.N, self.N))
        self.number_of_patterns_stored = 0
        self.last_training_r_state = {}

        # Create Matrices
        if W_sr is None:
            W_sr = spectral_radius

        self.W = self.create_W(sparseness=10/self.N, W_spectral_radius=W_sr)
        self.W_in = np.array(self.rng.normal(scale=W_in_std, size=(self.N, self.signal_dim)))
        self.W_out = np.array(self.rng.normal((self.signal_dim, self.N)))
        self.b = self.rng.normal(scale=b_std, size=self.N)

        self.r = np.zeros(self.N)

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

    def one_step_hallucinating(self, pattern_name=None):  # taken from page 113

        if self.number_of_patterns_stored == 0 or pattern_name is None:
            self.r = np.tanh(self.W @ self.r + self.D @ self.r + self.b)
        else:
            self.r = self.matrix_conceptor[pattern_name] @ np.tanh(self.W @ self.r + self.D @ self.r + self.b)

    def one_step_driving(self, pattern, pattern_name: str = None, noise_std=None):
        # print(np.shape(self.W_in))
        if noise_std:  # add noise to reservoir
            self.r = np.tanh(self.W @ self.r + self.W_in @ np.atleast_1d(pattern) +
                             self.b + np.random.normal(size=self.N, scale=noise_std))
        else:
            self.r = np.tanh(self.W @ self.r + self.W_in @ np.atleast_1d(pattern) + self.b)

    def set_r_state(self, pattern_name):
        self.r = self.last_training_r_state[pattern_name]

    def hallucinating(self, length, pattern_name: str = None, record_internal: bool = False, record_y: bool = False):
        self.set_r_state(pattern_name)

        recording_internal = []
        recording_y = []

        for t in range(length):
            if t%1000 == 0 and self.verbose:
                print(t)
            self.one_step_hallucinating(pattern_name)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(self.W_out @ self.r)
        return recording_internal, recording_y

    def record_r_z(self, pattern_name: str, n_washout: int, n_harvest: int, pattern, noise_std: float):
        record_r = []
        record_r_old = []
        record_p = []

        self.r = np.zeros(self.N)
        for t in range(n_washout):
            self.one_step_driving(pattern[t], pattern_name=pattern_name, noise_std=noise_std)

        for t in range(n_washout, n_washout + n_harvest):
            record_r_old.append(self.r)
            self.one_step_driving(pattern[t], pattern_name=pattern_name)
            record_r.append(self.r)
            record_p.append(np.atleast_1d(pattern[t]))
        if pattern_name not in self.last_training_r_state.keys():
            # print("added last training state to last_training_state")
            self.last_training_r_state[pattern_name] = self.r
        return record_r_old, record_r, record_p

    def compute_W_out_ridge(self, r_recordings, patterns, beta_W_out):
        y = np.array(np.vstack(patterns))
        X = np.vstack(r_recordings)
        ridge = Ridge(alpha=beta_W_out, fit_intercept=False)
        ridge.fit(X, y)
        W_out_optimized = ridge.coef_
        # print(f'W_out nrmse: {nrmse(W_out_optimized @ X.T, y.T)}')
        return W_out_optimized

    def compute_D_rigde(self, r_recordings, p_recordings, beta_D):
        p_stacked = np.vstack(p_recordings)
        y = np.array([self.W_in @ np.atleast_1d(p_t) for p_t in p_stacked])
        X = np.vstack(r_recordings)
        ridge = Ridge(alpha=beta_D, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        if self.verbose:
            print("D_optimized", np.shape(D_optimized))
        # print(f"nrmse D {nrmse(D_optimized @ X.T, np.array(y).T)}")
        return D_optimized

    def store_patterns(self, training_patterns, washout, n_adapt, beta_W, beta_W_out,
                       noise_std=None, signal_noise: float = None, **kwargs):
        self.training_patterns = training_patterns

        r_recordings = []
        p_recordings = []
        r_old_recordings = []
        matrix_collector = {}
        if signal_noise is not None:
            for key in training_patterns:
                noise = self.rng.normal(0, signal_noise, (len(training_patterns[key]), self.signal_dim))
                training_patterns[key] = training_patterns[key] + noise

        for name, training_pattern in training_patterns.items():
            # plt.plot(training_pattern[1000:1500,0], training_pattern[1000:1500,1], linewidth = 1)
            # plt.show()
            # print(name)
            # self.test_washout(washout=washout, pattern=training_pattern, num_neurons=3)
            record_r_old, record_r, record_p = self.record_r_z(pattern=training_pattern,
                                                           n_harvest=n_adapt,
                                                           n_washout=washout,
                                                           pattern_name=name,
                                                           noise_std=noise_std)
            r_old_recordings.append(record_r_old)
            r_recordings.append(record_r)
            p_recordings.append(record_p)
            record_r = np.array(record_r).T
            R = (record_r @ record_r.T) / n_adapt
            aperture = kwargs.get("aperture_" + name)
            # print(f"aperture {aperture}, beta_W: {beta_W}, beta_W_out: {beta_W_out}")
            self.matrix_conceptor[name] = R @ np.linalg.inv(R + (aperture ** -2) * np.identity(self.N))

            self.number_of_patterns_stored += 1
            s, v, d = np.linalg.svd(self.matrix_conceptor[name])
            # v = np.diagonal(v)
            # plt.plot(v)
            # plt.show()

        self.D = self.compute_D_rigde(r_old_recordings, p_recordings, beta_W)
        self.W_out = self.compute_W_out_ridge(r_recordings, p_recordings, beta_W_out)
        # self.print_NRMSEs(z_recordings, r_recordings, p_recordings)

    def record_chaotic(self, length, pattern_name):
        _, y_recording = self.hallucinating(length=length, pattern_name=pattern_name, record_internal=False,
                                            record_y=True)
        return y_recording

    def test_washout(self, washout, pattern, num_neurons):
        self.r = np.ones(self.N)
        self.plot_internal(washout, pattern, num_neurons)
        # print("here")
        self.r = np.zeros(self.N)
        self.plot_internal(washout, pattern, num_neurons)
        # plt.show()

    def plot_internal(self, washout, pattern, num_neurons):
        states = self.collect_internal(washout=washout, pattern=pattern)
        sampled_indices = range(num_neurons)
        sampled_rows = states[:, sampled_indices]
        # plt.figure(figsize=(10, 6))
        for row, id in zip(sampled_rows.T, sampled_indices):
            plt.plot(row, label=f"Neuron {id}")

        # Customize plot
        plt.title("Neurons values")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        # print(f"after this")

    def collect_internal(self, washout, pattern):
        # print(pattern)
        states = np.zeros(shape=(washout, self.N))
        for idx in range(washout):
            states[idx] = self.r
            self.one_step_driving(pattern=pattern[idx], pattern_name=None, noise_std=None)
        return states

