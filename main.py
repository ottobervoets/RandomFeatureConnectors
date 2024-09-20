from src.basic_resevoir_network import *
from src.signal_generators import *
from src.RFC_network import *

import matplotlib.pyplot as plt

N = 100


def plot_recording(recording: list[list[float]], neurons_to_plot: list[int]):
    for idx in neurons_to_plot:
        # Extract the values corresponding to this index from each vector
        values = [vec[idx] for vec in recording]
        print(values)
        plt.plot(values, label=f"Index {idx}")
    plt.show()

def test_basic_network():
    basic_network = BasicNetwork(100, non_zero=1)
    sinus_signal = sinus_discrete(n=0, period=10)
    print(sinus_signal)
    recordings = basic_network.driving_pattern(sinus_signal, record=True)
    recordings.extend(basic_network.hallucinate(200, record=True))
    values_first = [timestep[0] for timestep in recordings]
    print(values_first)
    plot_recording(recordings, range(0,5))


def main():
    rfc = RFCNetwork(N=100, M=500)
    n_adapt = 2000
    n_harvest = 400
    washout = 200
    learning_rate_c = 0.5
    beta_W_out = 0.01
    beta_D = 0.01
    appenture = 8

    patterns = []
    patterns.append(sinus_discrete(3000,8.83))
    patterns.append(sinus_discrete(3000, 9.83))
    patterns.append(random_pattern(3000,5))
    patterns.append(random_pattern(3000,4))

    rfc.store_patterns(patterns=patterns,
                       n_adapt=n_adapt,
                       washout=washout,
                       n_harvest = n_harvest,
                       beta_D = beta_D,
                       beta_W_out=beta_W_out)



if __name__ == "__main__":
    main()
