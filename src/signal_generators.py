import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(2)
def sinus_discrete(n, period=2 * np.pi):
    x = np.arange(0, n)
    y = np.sin(2 * x * np.pi / period)
    return y

def random_pattern(n, period):
    random_sequence = [np.random.uniform(low=-1, high=1) for _ in range(period)]
    random_sequence_1 = random_sequence[:-1]
    # Repeat the sequence enough times and slice it to get exactly 'n' numbers
    return (random_sequence * (n // period + 1))[:n]


def rossler_attractor(n, a=0.2, b=0.2, c=8.0, dt=1 / 200, subsample=150):
    # Initialize variables
    x, y, z = 0.0, 0.0, 0.0
    trajectory = []

    for i in range(n*subsample):
        # Compute derivatives
        dx = -(y + z)
        dy = x + a * y
        dz = b + x * z - c * z

        # Update variables using Euler's method
        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Store every subsample-th step in the trajectory
        if i % subsample == 0:
            trajectory.append([x, y])
    trajectory = np.array(trajectory)
    #scale to [0,1]
    scaled_trajectory = (trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory))

    return scaled_trajectory


def lorenz_attractor(n, step_size=1/200, subsample_rate=15, sigma=10, r=28, b=8/3):
    """
    Generate a normalized Lorenz attractor trajectory with specified length and parameters.

    Parameters:
    - n (int): Length of the final subsampled trajectory.
    - step_size (float): Step size for Euler approximation. Default is 1/200.
    - subsample_rate (int): Rate at which to subsample the trajectory. Default is 15.
    - sigma (float): Lorenz attractor parameter. Default is 10.
    - r (float): Lorenz attractor parameter. Default is 28.
    - b (float): Lorenz attractor parameter. Default is 8/3.

    Returns:
    - np.ndarray: A 2-dimensional array containing the normalized x and z coordinates.
    """
    # Compute the required total steps
    total_steps = n * subsample_rate

    # Initialize starting conditions
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []

    # Euler approximation over the calculated total steps
    for step in range(total_steps):
        # Calculate derivatives
        dx = sigma * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z

        # Update values using Euler method
        x += step_size * dx
        y += step_size * dy
        z += step_size * dz

        # Collect data at subsample intervals
        if step % subsample_rate == 0:
            trajectory.append([x, z])

    # Convert to numpy array for normalization
    trajectory = np.array(trajectory[:n])  # Ensure exactly n samples

    scaled_trajectory = (trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory))

    return scaled_trajectory


def mackey_glass(n=1000, step_size=1/10, beta=0.2, gamma=0.1, tau=17, exponent=10):
    """
    Generate a normalized Mackey-Glass time series with specified length and parameters.

    Parameters:
    - n (int): Length of the final time series.
    - step_size (float): Step size for Euler approximation. Default is 1/10.
    - beta (float): Mackey-Glass parameter. Default is 0.2.
    - gamma (float): Mackey-Glass parameter. Default is 0.1.
    - tau (int): Delay term for the Mackey-Glass equation. Default is 17.
    - exponent (int): Exponent in the Mackey-Glass equation. Default is 10.

    Returns:
    - np.ndarray: A 2-dimensional array with normalized pairs (x(t), x(t - tau)).
    """
    # Compute the number of total steps needed
    total_steps = n + tau

    # Initialize the time series with small random values for the warm-up period
    time_series = np.zeros(total_steps)
    time_series[0:tau] = 0.5   # Starting condition

    # Euler approximation to solve the delay differential equation
    for t in range(tau, total_steps - 1):
        x_tau = time_series[t - tau]
        dx = beta * x_tau / (1 + x_tau ** exponent) - gamma * time_series[t]
        time_series[t + 1] = time_series[t] + step_size * dx
        # print(time_series[t+1], dx)

    # Create the 2-dimensional time series (x(t), x(t - tau))
    pairs = np.array([[time_series[t], time_series[t - tau]] for t in range(tau, total_steps)])
    print(pairs.shape)
    # Take only the first `n` samples for the output
    pairs = pairs[:n]
    plt.plot(pairs[:,0], pairs[:,1])
    plt.show()

    # Normalize each channel to the range [0, 1]
    scaled_trajectory = (pairs - np.min(pairs)) / (np.max(pairs) - np.min(pairs))

    return scaled_trajectory

def mackey_glass_1(length, x0=None, a=0.2, b=0.1, c=10.0, tau=17,
                 n=1000, sample=0.46, discard=250):
    """Generate time series using the Mackey-Glass equation.

    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    time_series = x[n * discard::sample]
    pairs = np.array([[time_series[t], time_series[t - tau]] for t in range(tau, length)])
    scaled_trajectory = (pairs - np.min(pairs)) / (np.max(pairs) - np.min(pairs))


    # plt.plot(pairs[:,0], pairs[:,1])
    # plt.show()
    return scaled_trajectory


def henon_attractor(n=1000, a=1.4, b=0.3):
    """
    Generate a normalized Hénon attractor time series with specified length and parameters.

    Parameters:
    - n (int): Length of the time series.
    - a (float): Hénon attractor parameter. Default is 1.4.
    - b (float): Hénon attractor parameter. Default is 0.3.

    Returns:
    - np.ndarray: A 2-dimensional array with normalized pairs (x(n), y(n)).
    """
    # Initialize x and y arrays
    x, y = 0.0, 0.0
    trajectory = []

    # Iterate to generate the attractor sequence
    for _ in range(n):
        x_next = y + 1 - a * x**2
        y_next = b * x

        # Append current point to trajectory
        trajectory.append([x_next, y_next])

        # Update x and y for the next iteration
        x, y = x_next, y_next

    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)

    # Normalize each component separately to range [0, 1]
    scaled_trajectory = (trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory))

    return scaled_trajectory

