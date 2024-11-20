import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

class RFCNetwork:
    def __init__(self,
                 N,
                 M,
                 signal_dim=1,
                 spectral_radius=1.4,
                 lr_c=0.1,
                 aperture=4,
                 W_in_mean=0,
                 W_in_std=1.2,
                 b_mean=0,
                 b_std=0.2,
                 W_sparseness=0.1,
                 W_sr=None,
                 d_dim="sig_dim",
                 F_method="random",
                 G_method="random",
                 reproducible=False,
                 seed=294369130659753536483103517623731383366,
                 verbose = False,
                 **kwargs):

        if reproducible:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        self.verbose = verbose
        self.N = N
        self.M = M
        self.signal_dim = signal_dim
        self.aperture = aperture

        self.c = []
        self.number_of_patterns_stored = 0
        self.lr_c = lr_c

        # Create Matrices
        self.create_W(sparseness=W_sparseness, W_spectral_radius=W_sr)
        self.W_in = np.array(self.rng.normal(W_in_mean, W_in_std, (self.N, self.signal_dim)))
        self.W_out = np.array(self.rng.random((self.signal_dim, self.N)))
        self.b = self.rng.normal(b_mean, b_std, self.N)
        self.d_dim = d_dim
        self.G_tilde = self.rng.normal(0, 1, (self.N, self.M))

        self.create_D()
        self.create_F(F_method, G_method, kwargs)
        self.create_G(G_method)

        eigenvalues = np.linalg.eigvals(np.transpose(self.F) @ self.G)
        rho = np.max(np.abs(eigenvalues))
        a = np.power(spectral_radius / rho, 1 / 2)
        self.F = a * self.F
        self.G = a * self.G
        if self.verbose:
            print(f"spectral radius of F' G = {np.max(np.abs(np.linalg.eigvals(np.transpose(self.F) @ self.G)))} ")

        self.z_initial = self.rng.normal(0, 0.5, self.M)
        self.z = self.z_initial.copy()
        self.r = self.G @ self.z_initial.copy()

    def create_G(self, G_method):
        match G_method:
            case "W_G_tilde":
                self.G = self.W @ self.G_tilde
            case "W_F":
                self.G = self.W @ self.F
            case "random":
                self.G = self.G_tilde
            case "F":
                self.G = self.F.copy()
            case _:
                raise ValueError(f"{G_method} is not a supported way to create G. The methods are: W_G_tilde, " +
                                 f"W_F and random.")

    def create_W(self, sparseness=0.1, W_spectral_radius=None):
        total_elements = self.N ** 2
        num_non_zero = int(total_elements * sparseness)

        W_flat = np.zeros(total_elements)
        non_zero_indices = self.rng.choice(total_elements, num_non_zero, replace=False)
        W_flat[non_zero_indices] = self.rng.normal(0, 1, num_non_zero)

        W_sparse = W_flat.reshape(self.N, self.N)
        eigenvalues = np.linalg.eigvals(W_sparse)
        rho = np.max(np.abs(eigenvalues))
        if W_spectral_radius is None:
            self.W = W_sparse
        else:
            self.W = W_sparse * (W_spectral_radius / rho)
        if self.verbose:
            print("spectral radius of W:", np.max(np.abs(np.linalg.eigvals(self.W))))

    def create_F(self, F_method, G_method, kwargs):
        match F_method:
            case "random":
                self.F = np.array(self.rng.normal(0, 1, (self.N, self.M)))
            case "white_noise":
                self.construct_F_white_noise()
            case "patterns":
                if 'patterns' in kwargs:
                    self.construct_F_patterns(patterns=kwargs['patterns'])
                else:
                    raise ValueError("To construct F based on patterns, a list of patterns is needed.")
            case _:
                raise ValueError(
                    "Construction method not supported. Allowed methods: \"random\", \"white_noise\", \"patterns\"")
        if F_method != "random" and G_method == "random":
            raise Warning(
                "Making a non-random F matrix with a random G does not make sense. Performance is unpredicted")
        assert (np.isnan(self.F).any or np.isinf(self.F).any)

    def create_D(self):
        match self.d_dim:
            case "sig_dim":
                self.D = self.rng.normal(0, self.signal_dim, self.M)  # random initialize D
            case "reservoir_dim":
                self.D = self.rng.normal(0, 1, self.M)  # random initialize D
            case _:
                raise ValueError("D can either map back to the \"sig_dim\" or the \"reservoir_dim\" ")

    def construct_F_white_noise(self, sample_rate=50, washout=500):
        white_noise_sequence = self.rng.uniform(-1, 1, (self.M * sample_rate + washout, self.signal_dim))
        map_F = []
        self.r = self.rng.normal(0, 0.5, self.N)
        for idx in range(len(white_noise_sequence)):
            self.r = np.tanh(self.W @ self.r + self.W_in @ white_noise_sequence[idx])
            if idx >= washout and idx % sample_rate == 0:
                map_F.append(self.r)
        self.F = np.array(map_F).T
        if self.verbose:
            print("map F:", np.shape(self.F))

    def construct_F_patterns(self, patterns, washout=200):
        no_patterns = len(patterns)
        pattern_lens = [len(pattern) for pattern in patterns]
        sample_rates = [np.ceil((pattern_len - washout) / (self.M / no_patterns)) for pattern_len in pattern_lens]
        if any([sample_rate < 3 for sample_rate in sample_rates]):
            small_sample_rates = np.where(np.array(sample_rates) < 3)
            raise Warning(f"Sample rate of patterns{small_sample_rates} are below 3, give a longer sequence")

        map_F = []
        for pattern, sample_rate in zip(patterns, sample_rates):
            self.r = self.rng.normal(0, 1, self.N)
            for idx in range(len(pattern)):
                self.r = np.tanh(self.W @ self.r + self.W_in @ np.atleast_1d(pattern[idx]))
                if idx > washout and idx % sample_rate == 0:
                    map_F.append(self.r)
        while len(map_F) < self.M:
            map_F.append(self.rng.normal(0, 1, self.N))
            if self.verbose:
                print("Appended random vector to map F")
        self.F = np.array(map_F).T

    def __repr__(self):
        return f"RandomMatrixGenerator(N={self.N}, M={self.M}"

    def reset_r_z(self):
        self.z = self.z_initial.copy()  # right??
        self.r = self.G @ self.z

    def one_step_hallucinating(self, pattern_id=0):  # taken from page 113
        match self.d_dim:
            case "sig_dim":
                self.r = np.tanh(self.G @ self.z + self.W_in @ np.atleast_1d(self.D @ self.z) + self.b)
            case "reservoir_dim":
                self.r = np.tanh(self.G @ self.z + self.D @ self.z + self.b)
            case _:
                raise ValueError("Something wrong with d_dim")

        if self.number_of_patterns_stored == 0:
            self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
        else:
            self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r

    def hallucinating(self, time, pattern_id=0, record_internal=False, record_y=False, reset_resevoir = True):
        if reset_resevoir:
            self.reset_r_z()
        recording_internal = []
        recording_y = []
        for t in range(time):
            self.one_step_hallucinating(pattern_id)
            if record_internal:
                recording_internal.append(self.r)
            if record_y:
                recording_y.append(self.W_out @ self.r)
        return recording_internal, recording_y

    def c_adaptation(self, pattern, n_adapt, washout):

        self.c.append(np.ones(self.M))  # start at 1 conceptor
        # conceptor weight adaptation
        self.reset_r_z()
        for t in range(n_adapt):
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ np.tanh(
                self.G @ self.z + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
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
            self.r = np.tanh(self.G @ self.z + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ self.r

        for t in range(n_harvest):
            record_z.append(self.z)
            self.r = np.tanh(self.G @ self.z + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
            self.z = np.diag(self.c[self.number_of_patterns_stored]) @ np.transpose(self.F) @ self.r
            record_r.append(self.r)
            record_p.append(pattern[t])

        return record_r, record_z, record_p

    def compute_W_out_ridge(self, r_recordings, patterns, beta_W_out):
        y = np.array(np.hstack(patterns))
        X = np.vstack(r_recordings)
        ridge = Ridge(alpha=beta_W_out, fit_intercept=False)
        ridge.fit(X, y)

        W_out_optimized = ridge.coef_
        return W_out_optimized

    def compute_D_rigde(self, z_recordings, p_recordings, beta_D):
        match self.d_dim:
            case "sig_dim":
                y = np.array(np.hstack(p_recordings))
                X = np.vstack(z_recordings)
                ridge = Ridge(alpha=beta_D, fit_intercept=False)
                ridge.fit(X, y)
                D_optimized = ridge.coef_.T
            case "reservoir_dim":
                p_stacked = np.hstack(p_recordings)
                y = [self.W_in @ np.atleast_1d(p_t) for p_t in p_stacked]
                X = np.vstack(z_recordings)
                ridge = Ridge(alpha=beta_D, fit_intercept=False)
                ridge.fit(X, y)
                D_optimized = ridge.coef_
            case _:
                raise ValueError("Something went wrong with d_dim when computing D")
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

    def store_patterns(self, patterns, n_adapt, washout, n_harvest, beta_D, beta_W_out, beta_G,
                       noise_mean, noise_std):

        if noise_mean is not None and noise_std is not None:
            for i in range(len(patterns)):
                noise = self.rng.normal(noise_mean, noise_std, (len(patterns[i]), self.signal_dim))
                patterns[i] = patterns[i] + noise
        z_recordings = []
        r_recordings = []
        p_recordings = []

        for pattern in patterns:
            self.c_adaptation(pattern, n_adapt, washout)
            record_r, record_z, record_p = self.record_r_z(pattern=pattern, n_harvest=n_harvest, n_washout=washout)

            z_recordings.append(record_z)
            r_recordings.append(record_r)
            p_recordings.append(record_p)

            self.number_of_patterns_stored += 1


        self.D = self.compute_D_rigde(z_recordings, p_recordings, beta_D)
        self.W_out = self.compute_W_out_ridge(r_recordings, p_recordings, beta_W_out)
        self.G = self.compute_G(z_recordings, beta_G)
        self.print_NRMSEs(z_recordings, r_recordings, p_recordings)


    def drive_system(self, pattern, pattern_id):
        self.reset_r_z()
        for p_t in pattern:
            self.r = np.tanh(self.G @ self.z + self.W_in @ np.atleast_1d(p_t) + self.b)

            if self.number_of_patterns_stored == 0:
                self.z = np.identity(self.M) @ np.transpose(self.F) @ self.r
            else:
                self.z = np.diag(self.c[pattern_id]) @ np.transpose(self.F) @ self.r
    def record_chaotic(self, time, washout_pattern, pattern_id):
        self.drive_system(washout_pattern,pattern_id)

        _, y_recording = self.hallucinating(time=time,pattern_id=pattern_id, record_internal=False, record_y=True, reset_resevoir = False)
        return y_recording