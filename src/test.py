import numpy as np
import matplotlib.pyplot as plt

ls = [0,1,2,3,4]

print(ls[0:2], ls[2:4])


def test_washout(self, washout, pattern, num_neurons):
    self.r = np.ones(self.N)
    self.plot_internal(washout, pattern, num_neurons)
    self.z = np.zeros(self.M)
    self.r = np.zeros(self.N)
    self.plot_internal(washout, pattern, num_neurons)


def plot_internal(self, washout, pattern, num_neurons):
    states = self.collect_internal(washout=washout, pattern=pattern)
    sampled_indices = np.sort(np.random.choice(self.N, num_neurons, replace=False))
    sampled_rows = states[:, sampled_indices]
    plt.figure(figsize=(10, 6))
    for row, id in zip(sampled_rows.T, sampled_indices):
        plt.plot(row, label=f"Neuron {id}")

    # Customize plot
    plt.title("Neurons values")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def collect_internal(self, washout, pattern):
    print(pattern)
    states = np.zeros(shape=(washout, self.N))
    for idx in range(washout):
        self.one_step_driving(pattern=pattern[idx], pattern_name=None, noise_std=None)
        states[idx] = self.r
    return states
