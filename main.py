from src.basic_resevoir_network import *
from src.signal_generators import *
# from src.RFC_network import *
from src.support_functions import *
from src.RFC_network import *
import matplotlib.pyplot as plt
from src.RFC_network_2 import *
# from src.RFC_network_old import *

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

    try:
        signal_dim = len(patterns[0][0])
    except TypeError:
        signal_dim = 1

    rfc = RFCNetwork(N=100,
                      M=500, signal_dim=signal_dim,spectral_radius=spectral_radius, lr_c=learning_rate_c, aperture=aperture)

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
        shift, nmrse = find_optimal_phase_shift(patterns[pattern_id][0:20], result[200:], 400)
        if nmrse>0.1:
            plot_aligned_series_with_optimal_shift(patterns[pattern_id][0:20], result[200:], max_shift=480,
                                                   segment_range=(0, 600))
        optimal_nrmse.append(nmrse)
        optimal_shift.append(shift)

    return optimal_shift, optimal_nrmse


def main():
    patterns = []
    patterns.append(sinus_discrete(3000, 8.83))
    patterns.append(sinus_discrete(3000, 9.83))

    print(analysis(patterns))
    patterns = []
    patterns.append(rossler_attractor(3000))
    print(analysis(patterns, aperture=1))




def main_2():
    n_harvest = 400
    washout = 500
    learning_rate_c = 0.5
    beta_W_out = 0.01
    beta_G = 1
    beta_D = 0.01
    aperture = 8
    spectral_radius = 1.4
    N = 100
    M = 500
    W_mean = 0
    W_std = 1
    n_adapt = 2000

    patterns = []
    patterns.append(sinus_discrete(3000, 9.83))
    patterns.append(sinus_discrete(3000,8.83))
    patterns.append(random_pattern(3000,4))
    patterns.append(random_pattern(3000,5))


    rfc = RFCNetwork(N=N,
                     M=M,
                     signal_dim=1,
                     spectral_radius=spectral_radius,
                     lr_c=learning_rate_c,
                     aperture=aperture,
                     d_dim="resevoir_dim",
                     F_method="white_noise",
                     G_method = "W_F",
                     W_sr=1.5,
                     W_sparseness=0.1,
                     patterns=patterns)


    rfc.store_patterns(patterns=patterns,
                       n_adapt=n_adapt,
                       washout=washout,
                       n_harvest = n_harvest,
                       beta_D = beta_D,
                       beta_W_out=beta_W_out,
                       beta_G = beta_G)
    i = 0
    for conceptor in rfc.c:
        plt.plot(np.sort(conceptor), label=f"{i}")
        i += 1
    plt.legend()
    plt.show()

    for i in range(len(patterns)):
        _, result = rfc.hallucinating(800, i, False, True)

        plot_aligned_series_with_optimal_shift(patterns[i][0:20], result[400:], max_shift=299, segment_range=(0,600))
        plt.show()



    # plt.plot(internal[0], internal[1], "o")
    # plt.show()
    # plt.pause(3)
    # plt.close("all")
    # plt.imshow(rfc.D @ np.diag(rfc.c[1]) @ np.transpose(rfc.F))
    # plt.colorbar()
    # plt.show()
    # plot_internal(internal, range(0, 5), time=slice(0, 50))
    # for number in range(0,1001):
    #     plot_internal(internal, [number], 1)



if __name__ == "__main__":
    main_2()














