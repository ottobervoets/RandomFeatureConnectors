import numpy as np
import matplotlib.pyplot as plt


class MatrixConceptor:

    def __init__(self,
                 N: int = 250,
                 W_sr: float = 0.6,
                 W_in_std: float = 1.2,
                 bias: float = .2,
                 **kwargs):
        self.verbose = False
        self.N = N
        self.W_star = self.create_W(sparseness=10/self.N, W_spectral_radius=W_sr)
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
            if name == "lorenz_attractor" or name == "mackey_glass" or name == "lorenz_attractor_2d" or name == "mackey_glass_2d":
                values_in = (values_in * 2) - 1
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
                    P.append(values[t])
                    conceptor_recordings.append(self.r)

            conceptor_recordings = np.array(conceptor_recordings).T
            aperture = kwargs["aperture_" + name]
            self.construct_conceptor(aperture=aperture, xCollector=conceptor_recordings, name=name, n_adapt=n_adapt)

            self.last_training_state[name] = self.r
        X_tilde, X, P, B = np.array(X_tilde).T, np.array(X).T, np.array(P).T, np.array(B).T
        self.W_out = (np.linalg.inv(X @ X.T + beta_W_out * np.identity(self.N)) @ X @ P.T).T

        self.W = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W * np.identity(self.N)) @ X_tilde @ (np.arctanh(X) - B).T).T
        # self.D = (np.linalg.inv(X_tilde @ X_tilde.T + beta_W + np.identity(self.N)) @ X_tilde @ (self.W_in @ P).T).T

    def one_step_halucinating(self, pattern_name):
        self.r = self.C[pattern_name] @ np.tanh(self.W @ self.r + self.b)


    def record_chaotic(self, length, pattern_name):
        self.r = self.last_training_state[pattern_name]
        prediction = np.zeros((length, 2))
        for t in range(length):
            self.one_step_halucinating(pattern_name=pattern_name)
            prediction[t] = self.W_out @ self.r
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
        R = xCollector @ xCollector.T /n_adapt
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
