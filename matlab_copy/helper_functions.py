import numpy as np
from scipy.sparse import random as sprandn
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def generate_internal_weights(n_internal_units, connectivity):
    success = False
    while not success:
        try:
            # Generate sparse random matrix
            internal_weights = sprandn(n_internal_units, n_internal_units, density=connectivity, data_rvs=np.random.randn).toarray()
            # Compute spectral radius
            spec_rad = abs(eigs(internal_weights, k=1, which='LM', return_eigenvectors=False).max())
            internal_weights /= spec_rad  # Normalize to spectral radius of 1
            success = True
        except Exception:
            pass  # Retry on failure
    return internal_weights


def henon_attractor_2d(total_time, init_washout_length=1000):
    sample_length = total_time
    hs = np.array([1.2677, -0.0278]) #+ 0.01 * np.random.randn(size = 2)
    a, b = 1.4, 0.3
    henon_series = np.zeros((2, sample_length))

    # Washout phase
    for _ in range(init_washout_length):
        hs = np.array([hs[1] + 1 - a * hs[0] ** 2, b * hs[0]])

    # Generate sequence
    for n in range(sample_length):
        hs = np.array([hs[1] + 1 - a * hs[0] ** 2, b * hs[0]])
        henon_series[:, n] = hs

    # Normalize range to [0, 1]
    max_val = henon_series.max(axis=1)
    min_val = henon_series.min(axis=1)
    henon_series = (henon_series - min_val[:, None]) / (max_val - min_val)[:, None]

    return henon_series.T


def lorenz_attractor_2d(total_time, increments_per_unit=200, subsample_rate=15, init_washout_length=5000):
    sample_length = total_time
    ls = np.array([10.036677794959058, 9.98674414052542, 29.024692318601613]) + 0.01 * np.random.randn(3)
    sigma, b, r = 10.0, 8.0 / 3, 28.0
    delta = 1 / increments_per_unit
    lorenz_series = np.zeros((2, sample_length))

    # Washout phase
    for _ in range(init_washout_length):
        ls += delta * np.array([sigma * (ls[1] - ls[0]),
                                r * ls[0] - ls[1] - ls[0] * ls[2],
                                ls[0] * ls[1] - b * ls[2]])

    # Generate sequence
    for n in range(sample_length):
        for _ in range(subsample_rate):
            ls += delta * np.array([sigma * (ls[1] - ls[0]),
                                    r * ls[0] - ls[1] - ls[0] * ls[2],
                                    ls[0] * ls[1] - b * ls[2]])
        lorenz_series[:, n] = [ls[0], ls[2]]

    # Normalize range to [0, 1]
    max_val = lorenz_series.max(axis=1)
    min_val = lorenz_series.min(axis=1)
    lorenz_series = (lorenz_series - min_val[:, None]) / (max_val - min_val)[:, None]

    return lorenz_series.T



def mackey_glass_2d(total_time, tau=17, increments_per_unit=200, subsample_rate = 3, init_washout_length = 5000):
    sample_length = total_time
    gen_history_length = tau * increments_per_unit
    seed = 1.2 * np.ones(gen_history_length) + 0.2 * (np.random.rand(gen_history_length) - 0.5)
    old_val = 1.2
    gen_history = seed.copy()

    mg_series = np.zeros((2, sample_length))
    step = 0

    # Generate sequence
    for n in range(sample_length + init_washout_length):
        for _ in range(increments_per_unit * subsample_rate):
            step += 1
            tau_val = gen_history[step % gen_history_length]
            new_val = old_val + (0.2 * tau_val / (1.0 + tau_val ** 10) - 0.1 * old_val) / increments_per_unit
            gen_history[step % gen_history_length] = old_val
            old_val = new_val
        if n >= init_washout_length:
            mg_series[:, n - init_washout_length] = [new_val, tau_val]

    # Normalize range to [0, 1]
    max_val = mg_series.max(axis=1)
    min_val = mg_series.min(axis=1)
    mg_series = (mg_series - min_val[:, None]) / (max_val - min_val)[:, None]

    return mg_series.T


def rossler_attractor_2d(total_time, increments_per_unit=200, subsample_rate=150, init_washout_length=5000):
    sample_length = total_time
    rs = np.array([0.5943, -2.2038, 0.0260]) + 0.01 * np.random.randn(3)
    a, b, c = 0.2, 0.2, 8.0
    delta = 1 / increments_per_unit
    roessler_series = np.zeros((2, sample_length))

    # Washout phase
    for _ in range(init_washout_length):
        rs += delta * np.array([
            -(rs[1] + rs[2]),
            rs[0] + a * rs[1],
            b + rs[0] * rs[2] - c * rs[2]
        ])

    # Generate sequence
    for n in range(sample_length):
        for _ in range(subsample_rate):
            rs += delta * np.array([
                -(rs[1] + rs[2]),
                rs[0] + a * rs[1],
                b + rs[0] * rs[2] - c * rs[2]
            ])
        roessler_series[:, n] = [rs[0], rs[1]]

    # Normalize range to [0, 1]
    max_val = roessler_series.max(axis=1)
    min_val = roessler_series.min(axis=1)
    roessler_series = (roessler_series - min_val[:, None]) / (max_val - min_val)[:, None]

    return roessler_series.T


def nrmse(output, target):
    """
    Compute the normalized root mean square error (NRMSE) between the output and target.

    Parameters:
        output (numpy.ndarray): The output array (2D: features x samples).
        target (numpy.ndarray): The target array (2D: features x samples).

    Returns:
        numpy.ndarray: NRMSE values for each feature (1D: features).
    """
    # Compute the combined variance (mean of variances along features)
    combined_var = 0.5 * (np.var(target, axis=1, ddof=0) + np.var(output, axis=1, ddof=0))

    # Compute the error signal
    error_signal = output - target

    # Calculate NRMSE
    nrmse_values = np.sqrt(np.mean(error_signal**2, axis=1) / combined_var)

    return nrmse_values


if __name__ == "__main__":
    pattern_generators = [rossler_attractor_2d,  # these are those from herbert.
                          lorenz_attractor_2d,
                          mackey_glass_2d,
                          henon_attractor_2d]
    for gen in pattern_generators:
        patt = gen(total_time=2000)
        plt.plot(patt[:,0], patt[:,1], linewidth = 0.5)
        plt.show()
