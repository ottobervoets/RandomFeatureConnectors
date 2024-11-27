import matplotlib.pyplot as plt
import pandas as pd
from src.support_functions import *

import numpy as np
import csv
from src.defaultparms import default_parmas_chaotic, parameters_to_optimize, optimization_settings
from datetime import datetime
from models.factory import create_RFC
from signal_generators import rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor

def extract_lowest_nrmse_parameters(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Validate the structure of the CSV
    if "nrmse" not in df.columns:
        raise ValueError("CSV must contain a 'nrmse' column.")
    if len(df.columns) < 2:
        raise ValueError("CSV must contain at least one parameter column and the 'nrmse' column.")

    # Find the row with the lowest nrmse
    min_nrmse_row = df.loc[df["nrmse"].idxmin()]

    # Convert NaN values to None and create the dictionary
    result = {col: (None if pd.isna(min_nrmse_row[col]) else min_nrmse_row[col])
              for col in df.columns if col != "nrmse"}

    return result

def predict_choatic_systems(test_length=84, **best_params):

    pattern_generators = [rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor]
    # add training pattern to params.
    best_params['training_patterns'] = []
    true_patterns = [] #todo make dict, just like the starting values.
    training_length = best_params['n_adapt'] + best_params['washout']
    for pattern_generator in pattern_generators:
        total_pattern = pattern_generator(total_time=training_length + test_length)
        best_params['training_patterns'].append(total_pattern[0:training_length])
        true_patterns.append(total_pattern[training_length:test_length + training_length])
    rfc = create_RFC(**best_params)
    rfc.store_patterns(**best_params)
    result = {}
    for idp in range(len(true_patterns)):
        name = pattern_generators[idp].__name__
        predict = rfc.record_chaotic(test_length, pattern_id=idp)
        true = true_patterns[idp]
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
        if name == "henon_attractor":
            ax.scatter(true_x, true_y, label='True', color='blue', s=10)
            ax.scatter(predict_x, predict_y, label='Predicted', color='orange', s=10)
        else:
            ax.plot(true_x, true_y, label='True', color='blue', linestyle='-', marker='o', linewidth = 0.5, markersize=1)
            ax.plot(predict_x, predict_y, label='Predicted', color='orange', linestyle='-', marker='o', linewidth = 0.5, markersize=1)

        # Titles and annotations
        ax.set_title(name)
        ax.legend()
        ax.text(0.5, -0.1, f"NRMSE: {nrmse:.4f}", ha='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    if save_fig:
        if save_path is None:
            raise ValueError("Please provide a valid save_path when save_fig is True.")
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    path = "../res/"+"2024-11-27 09:33:19.csv"
    best_params = extract_lowest_nrmse_parameters(path)
    best_params['verbose'] = True

    best_params = default_parmas_chaotic

    results = predict_choatic_systems(test_length=84, **best_params)

    plot_predictions(results)





