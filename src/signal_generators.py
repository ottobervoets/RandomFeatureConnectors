import numpy as np

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