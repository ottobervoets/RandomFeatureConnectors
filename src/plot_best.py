import matplotlib.pyplot as plt
import pandas as pd
import cProfile
from src.support_functions import *

import numpy as np
import csv
from src.defaultparms import *
from datetime import datetime
from models.factory import create_RFC
from signal_generators import rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor
from matlab_copy.helper_functions import (rossler_attractor_2d,
                                          mackey_glass_2d,
                                          lorenz_attractor_2d,
                                          henon_attractor_2d)


def extract_lowest_nrmse_parameters(csv_file, M=None):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Validate the structure of the CSV
    if "nrmse" not in df.columns:
        raise ValueError("CSV must contain a 'nrmse' column.")
    if len(df.columns) < 2:
        raise ValueError("CSV must contain at least one parameter column and the 'nrmse' column.")

    # Find the row with the lowest nrmse
    if M is not None:
        df = df[df['M'] == M]
    min_nrmse_row = df.loc[df["nrmse"].idxmin()]


    # Convert NaN values to None and create the dictionary
    result = {col: (None if pd.isna(min_nrmse_row[col]) else min_nrmse_row[col])
              for col in df.columns if col != "nrmse"}

    return result

def predict_choatic_systems(test_length=84,  noise=None, **best_params):
    ############### DATA LOADING FROM HERBERT #############################
    data = {}
    # data['rossler_attractor'] = pd.read_csv("../data/RoesslerSeq.csv", header=None).to_numpy().T
    # data['lorenz_attractor'] = pd.read_csv("../data/lorenz.csv", header=None).to_numpy().T
    # data['mackey_glass'] = pd.read_csv("../data/MGSeq.csv", header=None).to_numpy().T
    # data['henon_attractor'] = pd.read_csv("../data/HenonSeq.csv", header=None).to_numpy().T
    data['rossler_attractor'] = np.loadtxt('../matlab_copy/1.csv', delimiter=",").T
    data['lorenz_attractor'] = 0.5 * (1 +np.loadtxt('../matlab_copy/2.csv', delimiter=",").T)
    data['mackey_glass'] = 0.5 * (1 +np.loadtxt('../matlab_copy/3.csv', delimiter=",").T)
    data['henon_attractor'] = np.loadtxt('../matlab_copy/4.csv', delimiter=",").T

    # pattern_generators = [lorenz_attractor]#, mackey_glass, henon_attractor]
    # pattern_generators = [rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor]
    pattern_generators = [rossler_attractor_2d,  # these are those from herbert.
                          lorenz_attractor_2d,
                          mackey_glass_2d,
                          henon_attractor_2d]
    # add training parttern rto params.
    best_params['verbose'] = True
    best_params['training_patterns'] = {}
    best_params['apperture'] = {}
    true_patterns = {}
    # training_length = np.max([best_params['n_adapt'] + best_params['washout'], best_params['n_adapt'] + best_params['sample_rate'] * best_params['M']])
    training_length = best_params['n_adapt'] + best_params['washout']
    total_pattern = {}
    for pattern_generator in pattern_generators:
        # total_pattern[pattern_generator.__name__] =  data[pattern_generator.__name__]#pattern_generator(total_time=training_length + test_length)
        total_pattern[pattern_generator.__name__] = pattern_generator(total_time=training_length + test_length)
        best_params['training_patterns'][pattern_generator.__name__] = total_pattern[pattern_generator.__name__][0:training_length]
        # true_patterns[pattern_generator.__name__] = total_pattern[pattern_generator.__name__][training_length-1:(training_length+test_length-1)]
        true_patterns[pattern_generator.__name__] = total_pattern[pattern_generator.__name__][training_length:(training_length+test_length)]

    rfc = create_RFC(**best_params)
    # print(rfc.training_patterns, "after construction")
    rfc.store_patterns(**best_params)
    # if noise is not None:
    #     for key, value in rfc.last_training_z_state.items():
    #         rfc.last_training_z_state[key] = value + np.random.normal(loc=0,scale=noise, size=best_params['M'])
    # print("#######",total_pattern["henon_attractor"][0] - rfc.training_patterns["henon_attractor"][0])
    # print("--------", np.shape(total_pattern["henon_attractor"]), np.shape(rfc.training_patterns["henon_attractor"]))
    # print("++++", rfc.training_patterns.keys(), total_pattern.keys())
    result = {}
    for idp in range(len(true_patterns)):
        name = pattern_generators[idp].__name__
        predict = rfc.record_chaotic(length=test_length, pattern_name=name)
        true = true_patterns[name]
        nmrse = NRMSE_2_dim(predict,
                            true)
        result[name] = {'true':true,
                        'predict':predict,
                        'nrmse':nmrse,
                        }
    return result


def plot_predictions(data, save_fig=False, save_path=None):
    """
    Plots predictions and true values from a dictionary where each item contains x and y in a list.

    Parameters:
        data (dict): Dictionary where keys are names, and values are dictionaries
                     with keys 'predict', 'true', and 'nrmse', where each is a list of [x, y] pairs.
        save_fig (bool): Whether to save the figure to a file. Default is False.
        save_path (str): Path to save the figure if save_fig is True. Default is None.
    """
    num_items = len(data)
    fig, axes = plt.subplots(1, num_items, figsize=(5 * num_items, 5))

    if num_items == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot.

    for ax, (name, values) in zip(axes, data.items()):
        true = values['true']
        predict = values['predict']
        nrmse = values['nrmse']

        # Separate x and y coordinates from the pairs of [x, y]
        true_x, true_y = zip(*true)
        predict_x, predict_y = zip(*predict)

        # Plotting the data
        if name == "henon_attractor" or name == "henon_attractor_2d":
            ax.scatter(true_x, true_y, label='True', color='blue', s=1)
            ax.scatter(predict_x, predict_y, label='Predicted', color='orange', s=1)
        else:
            ax.plot(true_x, true_y, label='True', color='blue', linestyle='-', marker='o', linewidth = 0.5, markersize=1)
            ax.plot(predict_x, predict_y, label='Predicted', color='orange', linestyle='-', marker='o', linewidth = 0.5, markersize=1)
        ax.plot(predict_x[0], predict_y[0], marker='o', color='Green')
        ax.plot(predict_x[-1], predict_y[-1], marker='o', color='Red')
        # Titles and annotations
        ax.set_title(name)
        ax.legend()
        ax.text(0.5, -0.1, f"NRMSE: {nrmse:.5f}", ha='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    if save_fig:
        if save_path is None:
            raise ValueError("Please provide a valid save_path when save_fig is True.")
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    matrix_parameters = {'N':500, 'W_sr': 0.6, 'bias':0.4, 'W_in_std': 1.2,
                         'beta_W': 0.0001, 'beta_W_out': 0.01,
                         'aperture_rossler_attractor': 3000,
                         'aperture_lorenz_attractor': 400,
                         'aperture_mackey_glass': 1300,
                         'aperture_henon_attractor': 630,
                         'n_adapt': 2000, 'washout':500, 'M':1,
                         'rfc_type': 'matrix_conceptor'


                         }



    # path = "../res/"+"2024-12-18 11_06_50.csv" #base
    path = "../res/" + "2025-01-24 11:34:43.csv" #pca
    # path = "../res/optimize_different_M_2/2025-01-06 16_26_39.csv"

    # path = "../res/" + "2024-12-18 11_03_39.csv"  # pca bigger
    # path = "../res/" + "2024-12-18 11_04_17.csv"  # base bigger
    best_params = extract_lowest_nrmse_parameters(path)#, M = 500)

    # best_params['aperture_rossler_attractor'] = 500
    # best_params = default_parmas_chaotic
    # best_params['verbose'] = True
    # best_params['sample_rate'] = 10

    # best_params['M'] = 125
    # best_params['N'] = 250
    # best_params['max_n_features'] = 250
    # best_params['aperture_mackey_glass'] = 10
    # best_params['rfc_type'] = 'PCARFC'
    # best_params = default_parmas_matrix

    # # noise = 0.01
    # print(best_params)
    # # best_params['aperture_rossler_attractor'] = 300
    for name, value in best_params.items():
        print(f"\'{name}\': {value},")
    # best_params['n_adapt'] = 3000
    # best_params['verbose'] = True
    # best_params['rfc_type'] = 'base'
    # best_params['aperture_lorenz_attractor'] =
    # best_params['aperture_henon_attractor'] = 50

    # best_params['signal_noise'] = 0.001

    # cProfile.run("predict_choatic_systems(test_length=84, **best_params)", sort="cumtime")
    # np.random.seed(1)
    best_params['n_adapt'] = 1900
    results = predict_choatic_systems(test_length=84, noise=None, **best_params)
    # results = predict_choatic_systems(test_length=84, noise=None, **default_parmas_chaotic)
    # results = predict_choatic_systems(test_length=500, noise=None, **best_params)
    # print(best_params['rfc_type'])
    plot_predictions(results)







