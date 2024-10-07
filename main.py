from src.basic_resevoir_network import *
from src.signal_generators import *
from src.RFC_network import *
from src.support_functions import *
from src.RFC_network_2 import *
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

def analysis(patterns,
             learning_rate_c = 0.5,
             aperture = 8,
             spectral_radius = 1.4,
             beta_W_out = 1,
             beta_G = 0.01,
             beta_D = 0.01):

    n_adapt = 2000
    n_harvest = 400
    washout = 200

    rfc = RFCNetwork(N=100, M=500, spectral_radius=spectral_radius, lr_c=learning_rate_c, aperture=aperture)

    rfc.store_patterns(patterns=patterns,
                       n_adapt=n_adapt,
                       washout=washout,
                       n_harvest=n_harvest,
                       beta_D=beta_D,
                       beta_W_out=beta_W_out,
                       beta_G=beta_G)
    retrieved_patterns = []
    optimal_nrmse = []
    optimal_shift = []
    for pattern_id in range(len(patterns)):
        _, result = rfc.hallucinating(700, pattern_id, False, True)
        retrieved_patterns.append(result[:200])
        shift, nmrse = find_optimal_phase_shift(patterns[pattern_id], result, 40)
        optimal_nrmse.append(nmrse)
        optimal_shift.append(shift)

    return optimal_shift, optimal_nrmse


def main_2():
    n_adapt = 2000
    n_harvest = 400
    washout = 200
    learning_rate_c = 0.5
    beta_W_out = 1
    beta_G = 0.01
    beta_D = 0.01
    aperture = 8
    spectral_radius = 1

    rfc = RFCNetwork(N=100, M=500, spectral_radius=spectral_radius, lr_c=learning_rate_c, aperture=aperture)

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
                       beta_W_out=beta_W_out,
                       beta_G = beta_G)

    internal_T, result = rfc.hallucinating(800, 0, True, True)


    internal = transpose_internal(internal_T)


    plot_aligned_series_with_optimal_shift(patterns[0], result[200:], max_shift=40, segment_range=(0,600))

    plt.show()
    plt.plot(internal[0], internal[1], "o")
    plt.show()
    # plt.pause(3)
    # plt.close("all")
    # plt.imshow(rfc.D @ np.diag(rfc.c[1]) @ np.transpose(rfc.F))
    # plt.colorbar()
    # plt.show()
    plot_internal(internal, range(0, 5), time=slice(0, 50))
    # for number in range(0,1001):
    #     plot_internal(internal, [number], 1)

def main():
    settings = {}
    apetures = []
    spectral_radius = []



if __name__ == "__main__":
    main_2()
