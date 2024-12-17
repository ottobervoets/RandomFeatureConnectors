from src.basic_reservoir_network import *
from src.signal_generators import *
# from src.RFC_network import *
from src.support_functions import *
import matplotlib.pyplot as plt
# from legacy.RFC_network_2 import *
from src.experiments import *
from src.defaultparms import default_parms
# from src.RFC_network_old import *
# from models.base_rfc import BaseRFC
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
    n_harvest = 2000
    washout = 500
    beta_W_out = 0.1
    beta_G = 0.1
    beta_D = 0.08
    aperture = 300
    spectral_radius = 0.5
    N = 100
    M = 500
    n_adapt = 2000
    W_sr = 0.5
    W_sparseness = 0.2
    d_dim = "reservoir_dim"
    F_method = "white_noise"  # random, white_noise, patter
    G_method = "W_F"  # random, F, W_F, W_G_tilde

    patterns = {}
    patterns["rossler_attractor"] = rossler_attractor(total_time=3000)
    patterns["lorenz_attractor"] = lorenz_attractor(total_time=3000)
    patterns["mackey_glass"] = mackey_glass(total_time=3000)
    patterns["henon_attractor"] = henon_attractor(total_time=3000)
    kwargs = {}
    kwargs["aperture_rossler_attractor"] = 700
    kwargs["aperture_lorenz_attractor"] = 350
    kwargs["aperture_mackey_glass"] = 200
    kwargs["aperture_henon_attractor"] = 20

    kwargs['training_patterns'] = patterns
    kwargs['noise_std'] = 0.001 #default = 0.001
    kwargs['max_n_features'] = N
    kwargs['washout'] = washout
    kwargs['n_adapt'] = n_adapt

    rfc = PCARFC(N=N,
                  M=M,
                  signal_dim=2,
                  spectral_radius=spectral_radius,
                  aperture=aperture,
                  W_sr=W_sr,
                  W_sparseness=W_sparseness,
                  verbose=True,
                  **kwargs)

    # rfc.test_washout(washout, pattern=patterns["henon_attractor"], num_neurons=10)

    rfc.store_patterns(n_harvest=n_harvest,
                       beta_D=beta_D,
                       beta_W_out=beta_W_out,
                       beta_G=beta_G,
                       **kwargs)
    # i = 0
    # for conceptor in rfc.c:
    #     plt.plot(np.sort(conceptor), label=f"{i}")
    #     i += 1
    # plt.legend()
    # plt.show()
    washout_pattern = 500
    prediciton_horizon = 84
    start_prediction = washout + n_harvest
    for key, value in patterns.items():
        result = rfc.record_chaotic(length=prediciton_horizon, pattern_name=key)
        result = np.array(result)
        nrsme = NRMSE_2_dim(patterns[key][start_prediction:start_prediction + prediciton_horizon], result)
        plt.plot(result[:,0], result[:,1], label="Simulated", linewidth = 0.5)
        plt.plot(patterns[key][start_prediction-1:start_prediction + prediciton_horizon, 0],
                 patterns[key][start_prediction-1:start_prediction + prediciton_horizon, 1], label="True", linewidth = 0.5)
        plt.plot(patterns[key][start_prediction, 0], patterns[key][start_prediction, 1], 'ro', markersize=2)
        plt.plot(result[0, 0], result[0, 1], 'go', markersize=2)
        # plt.plot(result[-1, 0], result[-1, 1], 'bo', markersize=1)

        plt.legend()
        plt.title(f"NRSME = {nrsme}, {"Good" if nrsme < 0.001 else "Bad"}")
        plt.show()



def main_1_dim():
    n_harvest = 800
    washout = 500
    beta_W_out = 0.01
    beta_G = 1
    beta_D = 0.01
    aperture = 8
    spectral_radius = 1.4
    N = 500
    M = 2500
    n_adapt = 2000
    W_sr = 1
    W_sparseness = 0.1


    patterns = {}
    patterns["sinus_1"] = sinus_discrete(3000, 9.83)
    patterns["sinus_2"] = sinus_discrete(3000, 8.83)
    patterns["discrete_1"] = random_pattern(3000, 4)
    patterns["discrete_2"] = random_pattern(3000, 5)

    extra_agrs = {"patterns": patterns, "n_adapt": n_adapt, "washout": washout, "max_n_components": 50}
    appertures = {"aperture_sinus_1": 8,
                  "aperture_sinus_2": 8,
                  "aperture_discrete_1": 8,
                  "aperture_discrete_2": 8,}
    rfc = PCARFC(N=N,
                  M=M,
                  signal_dim=1,
                  spectral_radius=spectral_radius,
                  aperture=aperture,
                  W_sr=W_sr,
                  W_sparseness=W_sparseness,
                  verbose=True,
                  training_patterns = patterns,
                 n_adapt = n_adapt,
                 washout = washout,
                 max_n_features = 100)

    rfc.store_patterns(training_patterns=patterns,
                       washout=washout,
                       n_harvest=n_harvest,
                       beta_D=beta_D,
                       beta_W_out=beta_W_out,
                       beta_G=beta_G,
                       **appertures)

    i = 0
    for conceptor in rfc.c.values():
        plt.plot(np.sort(conceptor), label=f"{i}")
        i += 1
    plt.legend()
    plt.show()
    for key, value in patterns.items():
        result = rfc.record_chaotic(80, key)

        # plot_aligned_series_with_optimal_shift(value[0:20], result, max_shift=200)
        plt.plot(result, label = "Predict")
        plt.plot(value[n_harvest+washout:n_harvest+washout+80], label = "True")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # main_1_dim()
    main_2_dim()
    # try:
    #     main_1_dim()
    # except Exception as e:
    #     print(e)

    # main_2_dim()

    # try:
    #     main_2_dim()
    # except Exception as e:
    #     print(e)
