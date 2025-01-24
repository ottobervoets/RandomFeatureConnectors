import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


class MatrixConceptor:

    def __init__(self,
                 N: int = 250,
                 W_sr: float = 0.6,
                 W_in_std: float = 1.2,
                 bias: float = .2,
                 verbose: bool = False,
                 **kwargs):
        self.verbose = verbose
        self.N = N
        self.W_star = self.create_W(sparseness=10/self.N, W_spectral_radius=W_sr)
        if self.verbose:
            print(f"Spectral radius of W_star: {np.max(np.abs(np.linalg.eigvals(self.W_star)))}")
        self.W = None
        self.W_in = np.random.normal(loc=0, scale=W_in_std, size=(self.N, 2))  # signal dimension
        self.b = np.random.normal(loc=0, scale=bias, size=self.N)

        self.r = np.random.normal(loc=0, scale=0.1, size=self.N)
        self.C = {}
        self.W_out = None
        self.last_training_state = {}

    def create_W(self, sparseness=0.1, W_spectral_radius=None):
        sparseness = 10/self.N
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
            print(f'Num non zero = {np.sum(W_sparse != 0) / (self.N ** 2)}')
        if W_spectral_radius is None:
            return W_sparse
        else:
            return W_sparse * (W_spectral_radius / rho)

    def one_step_driving(self, p_t, reservoir_noise = None):
        if reservoir_noise is None:
            self.r = np.tanh(self.W_in @ p_t + self.W_star @ self.r + self.b)
        else:
            self.r = np.tanh(self.W_in @ p_t + self.W_star @ self.r + self.b + np.random.normal(loc=0, scale=reservoir_noise, size=self.N))

    def store_patterns(self, training_patterns: dict, signal_noise: float = None, washout: int = None, n_adapt: int = None,
                      beta_W_out: float = 0.01, beta_W: float = 0.0001, noise_std: float = None, **kwargs):
        X_tilde = []
        X = []
        P = []
        B = []
        self.reservoir_recording = {}
        for name, values in training_patterns.items():
            values_in = values
            if name == "lorenz_attractor" or name == "mackey_glass":
                # print(f"changed values {name}")
                # values_in = (values_in * 2) - 1
                a = 0
            # plt.show()
            # plt.plot(values[:,0], values[:,1]) #uncomment for plotting input sequence
            self.r = np.zeros(self.N)
            conceptor_recordings =[]
            if signal_noise is not None:
                # print("added signal noise")
                values += np.random.normal(loc=0, scale=signal_noise, size=np.shape(values))
            # print(f"driving with {name} with shape {np.shape(values)}")
            for t in range(washout+n_adapt):
                x_old = self.r
                # self.one_step_driving(p_t = values[t], reservoir_noise=noise_std)
                self.r = np.tanh(self.W_in @ values_in[t] + self.W_star @ self.r + self.b)

                if t >= washout:
                    X_tilde.append(x_old)
                    X.append(self.r)
                    B.append(self.b)
                # if name == "lorenz_attractor" or name == "mackey_glass":
                #     P.append(0.5 * (values[t]+1))
                # else:
                #     P.append(values[t])
                    P.append(values[t])
                    conceptor_recordings.append(self.r)
            # end = len(np.array(X))
            # print(f"len X {end}")
            # for i in range(3):
            #     plt.plot(np.array(X_tilde).T[i, end-50:end], label=f'X_tilde {i}')  # uncomment for delay
            #     plt.plot(np.array(X).T[i, end-50:end], label=f'X {i}')
            # plt.legend()
            # plt.title(f"{name}")
            # plt.show()


            conceptor_recordings = np.array(conceptor_recordings).T
            self.reservoir_recording[name] = conceptor_recordings
            R = (conceptor_recordings @ conceptor_recordings.T)/n_adapt
            aperture = kwargs.get("aperture_" + name)
            self.C[name] = R @ np.linalg.inv(R + (aperture ** -2) * np.identity(self.N))
            C_temp = R @ np.linalg.inv(R + (aperture ** -2) * np.identity(self.N))
            self.construct_conceptor(aperture=aperture, xCollector=conceptor_recordings, name=name, n_adapt=n_adapt)
            # print(f"difference C's= {np.linalg.norm(self.C[name]-C_temp,  2)}")
            # U, S, Vh = np.linalg.svd(self.C[name])
            # plt.plot(S)
            # plt.title(f"Conceptor {name}")
            # plt.show()

            self.last_training_state[name] = self.r
        X_tilde, X, P, B = np.array(X_tilde).T, np.array(X).T, np.array(P).T, np.array(B).T
        # temp = X_tilde
        # X_tilde = X
        # X = temp
        # print(f"shapes, X_tilde: {X_tilde.shape}, X: {X.shape}, P: {P.shape}, B: {B.shape}")
        self.W_out = (np.linalg.inv(X @ X.T + beta_W_out * np.identity(self.N)) @ X @ P.T).T
        # self.W = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W * np.identity(self.N)) @ X_tilde @ (self.W_star @ X_tilde + self.W_in @ P).T).T

        self.W = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W * np.identity(self.N)) @ X_tilde @ (np.arctanh(X) - B).T).T

        # copyed from matlab code.
        Wtargets = (np.arctanh(X) - np.tile(self.b, (len(training_patterns.keys()) * n_adapt,1)).T)

        # Compute W
        W_3 = (np.linalg.inv(np.dot(X_tilde, X_tilde.T) +
                           beta_W * np.eye(self.N)) @ X_tilde @ Wtargets.T).T
        # self.W = W_3

        W_out_2 = (np.linalg.inv(X @ X.T + beta_W_out * np.identity(self.N)) @ X @ P.T).T
        # print(f"difference W's= {np.linalg.norm(W_3 - self.W, 2)}")
        # print(f"difference W_outs's= {np.linalg.norm(W_out_2 - self.W_out, 2)}")

        X_ridge, Y_ridge = X_tilde.T, (np.arctanh(X) - B).T
        r = Ridge(alpha=beta_W, fit_intercept=False)
        r.fit(X_ridge, Y_ridge)
        W_2 = r.coef_
        # self.W = W_3

        W_star = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W * np.identity(self.N)) @ X_tilde @ (np.arctanh(X) - B).T).T
        self.D = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W + np.identity(self.N)) @ X_tilde @ (self.W_in @ P).T).T
        # print(np.shape(self.D @ X_tilde), P.shape)
        # print(f"NRMSE D     {nrmse(self.D @ X_tilde, self.W_in @ P)} \n"
        #       # f"NRMSE W_2   {nrmse(W_star @ X_tilde + self.W_in @ P, W_2 @ X_tilde)}\n"
        #       f"NRMSE self.W{nrmse(self.W @ X_tilde, Wtargets):.6f} \n"
        #       # f"NRMSE W     {nrmse(W_star @ X_tilde + self.W_in @ P, self.W @ X_tilde)} \n"
        #       f"NRMSE W_out {nrmse(self.W_out @ X, P)}\n"
        #       f"n_adapt:    {n_adapt}")

        # print(f"difference D + W raw and W = {np.linalg.norm(self.D + self.W - W_temp, 2)}")


    def one_step_halucinating(self, pattern_name):

        self.r = self.C[pattern_name] @ np.tanh(self.W @ self.r + self.b)
        # self.r = np.tanh(self.W @ self.r + self.b)
        # self.r = self.C[pattern_name] @ np.tanh(self.W_star @ self.r + self.b + self.D @ self.r)
        # self.r = np.tanh(self.W_star @ self.r + self.b + self.D @ self.r)


    def record_chaotic(self, length, pattern_name):
        self.r = self.last_training_state[pattern_name]
        prediction = np.zeros((length, 2))
        record_internal = []
        for t in range(length):
            # prediction[t] = self.W_out @ self.r
            self.one_step_halucinating(pattern_name=pattern_name)
            record_internal.append(self.r)
            prediction[t] = self.W_out @ self.r
        # self.plot_resevoir(record_internal, 5, name=pattern_name, predict_len=length)
        return prediction

    def plot_resevoir(self, internal_predict, no_neurons, name, train_length = 100, predict_len = 84):
        internal_predict = np.array(internal_predict)
        for neuron in range(no_neurons):
            x = range(-train_length, predict_len, 1)
            y = np.concatenate((self.reservoir_recording[name][neuron, -train_length:], internal_predict.T[neuron,:predict_len]))
            plt.plot(x, y, label=f'neuron {neuron}')  # uncomment for delay
        plt.title(f"Reservoir states while driving and predicting {name}")
        plt.legend()
        plt.show()

    def construct_conceptor(self, aperture, xCollector, name, n_adapt):
        # Perform Singular Value Decomposition (SVD)
        Ux, Sx, Vx = np.linalg.svd(np.dot(xCollector, xCollector.T) / n_adapt)

        # Create a diagonal matrix of singular values
        diagSx = np.diag(Sx)

        # Optionally set small values in diagSx to zero (uncomment if needed)
        # diagSx[np.diag(diagSx) < 1e-6] = 0

        # Compute R
        R = np.dot(Ux, np.dot(diagSx, Ux.T))
        self.C[name] = R @ np.linalg.inv(R + (aperture ** -2) * np.identity(self.N))

def nrmse(predicted, actual):
    if predicted.shape != actual.shape:
        raise ValueError("The shapes of 'predicted' and 'actual' must be identical.")
    # print(np.shape((predicted - actual)**2))
    # Compute combinedVar
    combinedVar = 0.5 * (np.var(actual, axis=0, ddof=0) + np.var(predicted, axis=0, ddof=0))

    # Compute errorSignal
    errorSignal = predicted - actual

    # Compute NRMSE
    NRMSE = np.sqrt(np.mean(errorSignal ** 2, axis=0) / combinedVar)
    # print(f"NRMSE book: {np.sqrt(np.mean((predicted - actual)**2)/np.mean(actual**2))} \n"
    #       f"NRMSE code: {np.mean(NRMSE)}")
    # return np.sqrt(np.mean((predicted - actual)**2)/np.mean(actual**2))
    return np.mean(NRMSE)
