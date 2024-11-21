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


def rossler_attractor(n=1000, a=0.2, b=0.2, c=8.0, dt=1 / 200, subsample=150):
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
    print(len(scaled_trajectory))
    return scaled_trajectory


def lorenz_attractor(n=1000, step_size=1/200, subsample_rate=15, sigma=10, r=28, b=8/3):
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
    print(len(scaled_trajectory))
    return scaled_trajectory

def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=17, dt=0.1, total_time=2500, subsample_rate = 10,normalize=True):

    tau_dt = int(tau/dt)
    washout = max(tau_dt, 5000)
    x = np.ones(total_time*subsample_rate + washout + tau_dt)*1.2
    time_series = []
    for t in range(tau_dt,len(x)):
        x[t] = x[t-1] + dt * (beta * x[t-1-tau_dt]/(1+x[t-1-tau_dt]**n) - gamma * x[t-1])
        if t%subsample_rate == 0 and t>=(washout+tau_dt):
            time_series.append([x[t],x[t-tau_dt]])

    time_series = np.array(time_series)
    # Normalize to [0, 1] range if required
    if normalize:
        time_series = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series))
    print(len(time_series))
    return time_series

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
    print(len(scaled_trajectory))

    return scaled_trajectory

if __name__ == "__main__":
    functions = [rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor]
    # functions = [mackey_glass, henon_attractor]
    for func in functions:
        results = func()
        if func.__name__ == "henon_attractor":
            plt.plot(results[:, 0], results[:, 1], 'o', markersize=1)
        else:
            plt.plot(results[:,0], results[:,1], linewidth = 0.5)
        plt.show()