import numpy as np


def sinus_discrete(n, period=2 * np.pi):
    x = np.arange(0, n)
    y = np.sin(2 * x * np.pi / period)
    return y

def random_pattern(n, period):
    random_sequence = [np.random.uniform(low=-1, high=1) for _ in range(period)]

    # Repeat the sequence enough times and slice it to get exactly 'n' numbers
    return (random_sequence * (n // period + 1))[:n]
