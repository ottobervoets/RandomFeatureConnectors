from src.basic_reservoir_network import *
from src.signal_generators import *
# from src.RFC_network import *
from src.support_functions import *
import matplotlib.pyplot as plt
# from legacy.RFC_network_2 import *
from src.experiments import *
from src.defaultparms import default_parms
# from src.RFC_network_old import *
from models.base_rfc import BaseRFC
from models.PCA_rfc import PCARFC
from models.randomG_rfc import RandomGRFC
import csv

N = 100


def main_experiments():
    n_reps = 10
    filename = 'res/experiment_1.csv'
    experiment_list = [
        {'F_method': 'random',
         'G_method': 'random'},
        {'F_method': 'random',
         'G_method': 'F'},
        {'F_method': 'random',
         'G_method': 'W_F'},  # should be same result (also random)
        {'F_method': 'white_noise',
         'G_method': 'W_F'},
        {'F_method': 'white_noise',
         'G_method': 'W_G_tilde'},
        {'F_method': 'patterns',
         'G_method': 'W_F'},
        {'F_method': 'patterns',
         'G_method': 'W_G_tilde'},
    ]
    patterns = []
    patterns.append(sinus_discrete(3000, 9.83))
    patterns.append(sinus_discrete(3000, 8.83))
    patterns.append(random_pattern(3000, 4))
    patterns.append(random_pattern(3000, 5))

    for experiment in experiment_list:
        params = {**default_parms, **experiment}
        try:
            mean, std = experiments(patterns, n_reps, **params)
        except Exception as e:
            print(f"Exeption {e} occured, going to next experiment")
            continue
        print(f"For experiment with settings: {params}, we have the following NRMSE{mean}")
        for i in range(len(mean)):
            params[f"pattern {i} mean"] = std[i]
            params[f"pattern {i} std"] = mean[i]
        params["n_rep"] = n_reps

        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=params.keys())

            # Write the header only if the file is new
            if not file_exists:
                writer.writeheader()  # Write the header (keys)

            writer.writerow(params)  # Write the data (values)


def main_2_dim():
    n_harvest = 1000
    washout = 500
    learning_rate_c = 0.5
    beta_W_out = 0.01
    beta_G = 0.5
    beta_D = 0.0001
    aperture = 300
    spectral_radius = 1.4
    N = 200
    M = 2000
    n_adapt = 2000
    W_sr = 1.2
    W_sparseness = 0.3
    d_dim = "reservoir_dim"
    F_method = "white_noise"  # random, white_noise, patter
    G_method = "W_F"  # random, F, W_F, W_G_tilde

    patterns = []
    patterns.append(rossler_attractor(3000))
    # patterns.append(lorenz_attractor(3000))
    # patterns.append(mackey_glass_1(3000))
    # patterns.append(henon_attractor(3000))

    rfc = BaseRFC(N=N,
                  M=M,
                  signal_dim=2,
                  spectral_radius=spectral_radius,
                  aperture=aperture,
                  W_sr=W_sr,
                  W_sparseness=W_sparseness,
                  verbose=True)

    rfc.store_patterns(patterns=patterns,
                       washout=washout,
                       n_harvest=n_harvest,
                       beta_D=beta_D,
                       beta_W_out=beta_W_out,
                       beta_G=beta_G)
    # i = 0
    # for conceptor in rfc.c:
    #     plt.plot(np.sort(conceptor), label=f"{i}")
    #     i += 1
    # plt.legend()
    # plt.show()
    washout_pattern = 500
    prediciton_horizon = 84
    start_prediction = washout + n_harvest
    for i in range(len(patterns)):
        result = rfc.record_chaotic(length=prediciton_horizon, pattern_id=i)
        result = np.array(result)
        plt.plot(result[:,0], result[:,1], label="Simulated")
        plt.plot(patterns[i][start_prediction:start_prediction + prediciton_horizon, 0],
                 patterns[i][start_prediction:start_prediction + prediciton_horizon, 1], label="True")
        plt.plot(result[0, 0], result[0, 1], 'ro')
        plt.legend()
        plt.show()


def main_1_dim():
    n_harvest = 800
    washout = 500
    learning_rate_c = 0.5
    beta_W_out = 0.01
    beta_G = 1
    beta_D = 0.01
    aperture = 8
    spectral_radius = 1.4
    N = 100
    M = 500
    n_adapt = 2000
    W_sr = 1.5
    W_sparseness = 0.1
    d_dim = "reservoir_dim"
    F_method = "patterns"
    G_method = "W_F"

    patterns = []
    patterns.append(sinus_discrete(3000, 9.83))
    patterns.append(sinus_discrete(3000, 8.83))
    patterns.append(random_pattern(3000, 4))
    patterns.append(random_pattern(3000, 5))

    extra_agrs = {"patterns": patterns, "n_adapt": n_adapt, "washout": washout, "max_n_components": 50}

    rfc = RandomGRFC(N=N,
                  M=M,
                  signal_dim=1,
                  spectral_radius=spectral_radius,
                  aperture=aperture,
                  W_sr=W_sr,
                  W_sparseness=W_sparseness,
                  verbose=True,
                  patterns = patterns,
                 n_adapt = n_adapt,
                 washout = washout,
                 max_n_components = 50)

    rfc.store_patterns(patterns=patterns,
                       washout=washout,
                       n_harvest=n_harvest,
                       beta_D=beta_D,
                       beta_W_out=beta_W_out,
                       beta_G=beta_G)

    i = 0
    for conceptor in rfc.c:
        plt.plot(np.sort(conceptor), label=f"{i}")
        i += 1
    plt.legend()
    plt.show()

    for i in range(len(patterns)):
        _, result = rfc.hallucinating(800, i, False, True)

        plot_aligned_series_with_optimal_shift(patterns[i][0:20], result[400:], max_shift=299, segment_range=(0, 600))
        plt.show()


if __name__ == "__main__":
    main_2_dim()
    #
    # try:
    #     main_1_dim()
    # except Exception as e:
    #     print(e)

    # main_2_dim()

    # try:
    #     main_2_dim()
    # except Exception as e:
    #     print(e)
