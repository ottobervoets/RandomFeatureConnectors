##########################
# this version has a lot of plot/test functions that make it less clear what is happening. but is great for plotting/checking stuff






import numpy as np
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt


class MatrixConceptor:
    def __init__(self,
                 washout: int = 500,
                 N: int = 500,
                 beta_W=0.0001,
                 verbose: bool = False,
                 W_std=1.5,
                 W_in_std=1.2,
                 bias=.4,
                 beta_W_out=0.01,
                 W_sr=0.6,
                 **kwargs
                 ):
        self.N = N
        self.beta_W = beta_W
        self.verbose = verbose
        self.washout = washout

        self.r = np.zeros(self.N)
        # self.r = np.random.normal(size=(self.N))
        self.W = self.create_W(W_spectral_radius=W_sr)
        if self.verbose:
            print(f"Spectral radius of W is now {np.max(np.abs(np.linalg.eigvals(self.W)))}")
        self.W_in = np.random.normal(scale=W_in_std, size=(N, 2))
        self.W_out = None
        self.beta_W_out = beta_W_out
        self.b = np.random.normal(scale=bias, size=N)
        self.D = np.zeros(shape=(self.N, self.N))

        self.C = {}  # store matrix Conceptors
        self.last_state = {}

    def create_W(self, sparseness=0.1, W_spectral_radius=None):
        total_elements = self.N ** 2
        num_non_zero = int(total_elements * sparseness)

        W_flat = np.zeros(total_elements)
        non_zero_indices = np.random.choice(total_elements, num_non_zero, replace=False)
        W_flat[non_zero_indices] = np.random.normal(0, 1, num_non_zero)

        W_sparse = W_flat.reshape(self.N, self.N)
        eigenvalues = np.linalg.eigvals(W_sparse)
        rho = np.max(np.abs(eigenvalues))
        if self.verbose:
            print("spectral radius of W:", np.max(np.abs(np.linalg.eigvals(W_sparse * (W_spectral_radius / rho)))))
            print(f'Num non zero = {np.sum(W_sparse != 0)/(self.N**2)}')
        if W_spectral_radius is None:
            return W_sparse
        else:
            return W_sparse * (W_spectral_radius / rho)

    def adjust_spectral_radius(self, spectral_radius, input_matrix):
        eigenvalues = np.linalg.eigvals(input_matrix)
        rho = np.max(np.abs(eigenvalues))
        a = spectral_radius / rho
        return a * input_matrix

    def one_step(self,
                 p_t: list[float] = None,
                 pattern_name: str = None,
                 noise_std: float = None):

        if p_t is not None:
            self.r = self.W @ self.r + self.b + self.W_in @ p_t
        else:
            self.r = self.W @ self.r + self.b #+ self.D @ self.r
        if noise_std is not None:
            self.r += np.random.normal(loc=0, scale=noise_std, size=self.N)

        self.r = np.tanh(self.r)

        if pattern_name is not None:
            self.r = self.C[pattern_name] @ self.r

    def construct_conceptors_direct(self,
                                    reservoir_recordings: dict,
                                    **kwargs):

        for name, value in reservoir_recordings.items():
            aperture = kwargs.get("aperture_" + name)
            if self.verbose:
                print(f"Aperture for {name} is {aperture}")
            R = reservoir_recordings[name].T @ reservoir_recordings[name] / len(reservoir_recordings[name])
            self.C[name] = R @ np.linalg.inv(R + aperture ** -2 * np.identity(self.N))

    def store_patterns(self,
                       training_patterns: dict,
                       n_adapt: int = 3000,
                       signal_noise: float = None,
                       noise_std: float = None,
                       **kwargs
                       ):
        X = []
        X_tilde = []
        P = []
        B = []
        reservoir_recordings = {}
        for name, values in training_patterns.items():
            self.r = np.zeros(self.N)
            if signal_noise is not None:
                training_patterns[name] = values + np.random.normal(0, signal_noise, size=(len(values), 2))
                values +=
            for t in range(self.washout):
                self.one_step(p_t=values[t], noise_std=noise_std)
            x = []
            p = []
            b = []
            for t in range(self.washout, self.washout + n_adapt):
                self.one_step(p_t=values[t])
                x.append(self.r)
                p.append(values[t])
                b.append(self.b)
            reservoir_recordings[name] = np.array(x)
            X.extend(x[1:])
            P.extend(p[1:])
            B.extend(b[1:])
            X_tilde.extend(x[:-1])

            self.last_state[name] = self.r
            if self.verbose:
                print(f"len(X): {len(X)}, len(P): {len(P)}, len(X_tilde): {len(X_tilde)}")

        # if self. verbose:
        #     for idx in range(len(training_patterns.values())):
        #         print(idx, np.shape(X))
        #
        #         data = np.array(X[idx * (n_adapt-1):(idx+1) * (n_adapt-1)])
        #         print(data.shape)
        #         data = data.T @ data / (n_adapt-1)
        #         _, singvals, _ = np.linalg.svd(data)
        #
        #         # plt.plot(-np.sort(-singvals)[0:10])
        #         # plt.show()

        X = np.array(X).T
        P = np.array(P).T
        X_tilde = np.array(X_tilde).T
        B = np.array(B).T
        W_old = self.W
        self.W_out = (np.linalg.inv(X @ X.T + self.beta_W_out * np.identity(self.N)) @ X @ P.T).T
        self.W = (np.linalg.inv(X_tilde @ X_tilde.T + self.beta_W * np.identity(self.N)) @ X_tilde @ (
                np.arctanh(X) - B).T).T
        ridge_X = X_tilde.T
        ridge_Y = [W_old @ x_temp + self.W_in @ p_t for x_temp, p_t in zip(X_tilde.T, P.T)] #+ [self.W_in @ p_t for p_t in P.T]
        ridge = Ridge(alpha=self.beta_W, fit_intercept=False)
        ridge.fit(ridge_X, ridge_Y)
        print(f"Difference both methods = {np.linalg.norm(ridge.coef_ - self.W, 2)}")
        # self.construct_D(P=P, R=X_tilde)
        self.W = ridge.coef_
        # print NRMSE
        W_error, W_out_error, D_error = 0, 0, 0
        X = X.T
        X_tilde = X_tilde.T
        P = P.T



        for idx in range(len(X)):
            W_error += np.mean(((np.arctanh(X[idx])) - self.W @ X_tilde[idx]) ** 2)
            # W_error += np.mean((self.W @ X_tilde[idx] - W_old @ X_tilde[idx] + self.W_in @ P[idx])**2)
            W_out_error += np.mean((P[idx] - self.W_out @ X[idx]))
            D_error += np.mean((self.D @ X[idx] - self.W_in @ P[idx])**2)
        if self.verbose:
            print(f"The error of W is {(W_error**0.5) / len(X)}, and {(W_out_error**0.5) / len(X)} for W_out, and {D_error/len(X)} for D")

        self.construct_conceptors_direct(reservoir_recordings=reservoir_recordings, **kwargs)
        # self.construct_conceptors(patterns=patterns, apertures=apertures)

    def record_chaotic(self, length, pattern_name):
        self.r = self.last_state[pattern_name]
        prediction = np.zeros((length, 2))
        for t in range(length):
            self.one_step(pattern_name=pattern_name)
            prediction[t] = self.W_out @ self.r
        return prediction

    def construct_D(self, P, R):
        y = [self.W_in @ np.atleast_1d(p_t) for p_t in P.T]
        X = R.T
        ridge = Ridge(alpha=self.beta_W, fit_intercept=False)
        ridge.fit(X, y)
        D_optimized = ridge.coef_
        self.D = D_optimized


    # def construct_conceptors(self,
    #                          patterns: dict,
    #                          apertures: dict):
    #     for name, values in patterns.items():
    #         # Washout
    #         for t in range(self.washout):
    #             self.one_step(pattvalues[t])
    #         collected_r = np.zeros((len(values) - self.washout, self.N))
    #         # Collect states
    #         for t in range(self.washout, len(values)):
    #             self.one_step(values[t])
    #             collected_r[t - self.washout] = self.r
    #             self.last_state[name] = self.r
    #
    #         # Compute conceptor and store
    #         R = collected_r.T @ collected_r / len(collected_r)
    #         self.C[name] = R @ np.linalg.inv(R + (apertures[name] ** -2) * np.identity(self.N))
