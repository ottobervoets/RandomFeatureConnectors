from src.RFC_network import RFCNetwork
from src.support_functions import *

import matplotlib.pyplot as plt
import numpy as np
import csv
from src.defaultparms import *
from datetime import datetime



def one_experiment(patterns, **kwargs):
    defaults = {
        'n_harvest': 400,
        'washout': 500,
        'learning_rate_c': 0.5,
        'beta_W_out': 0.01,
        'beta_G': 1,
        'beta_D': 0.01,
        'aperture': 8,
        'spectral_radius': 1.4,
        'N': 100,
        'M': 500,
        'n_adapt': 2000,
        'W_sr': 1.5,
        'W_sparseness': 0.1,
        'd_dim': "reservoir_dim",
        'F_method': "random",
        'signal_dim': 1,
        'G_method': "random",
        'noise_mean': None,
        'noise_std': None}
    unexpected_keys = set(kwargs) - defaults.keys()
    if unexpected_keys:
        raise ValueError(f"Unexpected parameter(s): {', '.join(unexpected_keys)}")

    params = {**defaults, **kwargs}

    rfc = RFCNetwork(N=params['N'],
                     M=params['M'],
                     signal_dim=params['signal_dim'],
                     spectral_radius=params['spectral_radius'],
                     lr_c=params['learning_rate_c'],
                     aperture=params['aperture'],
                     d_dim=params['d_dim'],
                     F_method=params['F_method'],
                     G_method=params['G_method'],
                     W_sr=params['W_sr'],
                     W_sparseness=params['W_sparseness'],
                     patterns=patterns)

    rfc.store_patterns(patterns=patterns,
                       n_adapt=params['n_adapt'],
                       washout=params['washout'],
                       n_harvest=params['n_harvest'],
                       beta_D=params['beta_D'],
                       beta_W_out=params['beta_W_out'],
                       beta_G=params['beta_G'],
                       noise_mean=params['noise_mean'],
                       noise_std=params['noise_std'])
    optimal_nrmses = []
    optimal_shifts = []

    for pattern_id in range(len(patterns)):
        _, result = rfc.hallucinating(700, pattern_id, False, True)
        shift, nmrse = find_optimal_phase_shift(patterns[pattern_id][0:20], result[200:], 400)
        if nmrse > 0.5:
            print(f"NMRSE large{nmrse}")
            # plot_aligned_series_with_optimal_shift(patterns[pattern_id][0:20], result[200:], max_shift=480,
            #                                        segment_range=(0, 600))
        optimal_nrmses.append(nmrse)
        optimal_shifts.append(shift)

    return optimal_nrmses, optimal_shifts


def experiments(patterns,
                n_reps,
                **kwargs):
    results = []
    for experiment in range(n_reps):
        nrmses, _ = one_experiment(patterns, **kwargs)
        results.append(nrmses)
    return np.mean(results, axis=0), np.std(results, axis=0)


def one_experiment_chaotic(patterns, washout_signal=500, simulation_period=84, **kwargs):
    defaults = {
        'n_harvest': 400,
        'washout': 500,
        'learning_rate_c': 0.5,
        'beta_W_out': 0.01,
        'beta_G': 1,
        'beta_D': 0.01,
        'aperture': 8,
        'spectral_radius': 1.4,
        'N': 100,
        'M': 500,
        'n_adapt': 2000,
        'W_sr': 1.5,
        'W_sparseness': 0.1,
        'd_dim': "reservoir_dim",
        'F_method': "random",
        'signal_dim': 1,
        'G_method': "random",
        'noise_mean': None,
        'noise_std': None}
    unexpected_keys = set(kwargs) - defaults.keys()
    # if unexpected_keys:
    #     raise ValueError(f"Unexpected parameter(s): {', '.join(unexpected_keys)}")

    params = {**defaults, **kwargs}

    rfc = RFCNetwork(N=params['N'],
                     M=params['M'],
                     signal_dim=params['signal_dim'],
                     spectral_radius=params['spectral_radius'],
                     lr_c=params['learning_rate_c'],
                     aperture=params['aperture'],
                     d_dim=params['d_dim'],
                     F_method=params['F_method'],
                     G_method=params['G_method'],
                     W_sr=params['W_sr'],
                     W_sparseness=params['W_sparseness'],
                     patterns=patterns)

    rfc.store_patterns(patterns=patterns,
                       n_adapt=params['n_adapt'],
                       washout=params['washout'],
                       n_harvest=params['n_harvest'],
                       beta_D=params['beta_D'],
                       beta_W_out=params['beta_W_out'],
                       beta_G=params['beta_G'],
                       noise_mean=params['noise_mean'],
                       noise_std=params['noise_std'])
    nrmses = []

    for idp in range(len(patterns)):
        result = rfc.record_chaotic(700, washout_pattern=patterns[idp][0:washout_signal], pattern_id=idp)
        nmrse = NRMSE_2_dim(patterns[idp][washout_signal:washout_signal + simulation_period],
                            result[0:simulation_period])
        if nmrse > 0.5:
            print(f"NMRSE large{nmrse}")

        nrmses.append(nmrse)

    return nrmses


def n_experiments_chaotic(patterns, n_reps, **kwargs):
    results = []
    d_params = {
        'washout_signal': 500,
        'simulation_period': 84,
    }
    params = {**d_params, **kwargs}
    for experiment in range(n_reps):
        nrmses = one_experiment_chaotic(patterns, washout_signal=params['washout_signal'],
                                        simulation_period=params['simulation_period'],
                                        **kwargs)
        results.append(nrmses)
    return np.mean(results, axis=0)

def optimize_parameters_chaotic(patterns, parameters_to_optimize, **kwargs):
     # get starting parameters
    optimized_params = {**default_parms,  **kwargs.get("params", {})}

    default_settings = {
        'max_iterations':10,
        'experiment_name': "res/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv",
        'repetitions':None,
        'n_rep':1
    }

    settings = {**default_settings, **kwargs}
    for _ in range(settings['repetitions']):
        for current_parameter, start_value in parameters_to_optimize.items():
            optimized_params[current_parameter] = start_value
            try: optimized_params['nrmse'] = n_experiments_chaotic(patterns, settings['n_rep'], **optimized_params)
            except Exception as e:
                print(e)

            counter = 0

            left_params = optimized_params
            right_params = optimized_params
            left_params['nrmse'] = 0
            right_params['nrmse'] = 0

            while optimized_params['nrmse'] > min(left_params['nrmse'], right_params['nrmse']) or counter != settings['max_iterations']:
                counter += 1
                left_params = optimized_params
                left_params[current_parameter] = left_params[current_parameter] * 0.5
                try: left_params['nrmse'] = n_experiments_chaotic(patterns, settings['n_rep'], **left_params)
                except Exception as e:
                    print(e)
                    break
                write_experiment_results(left_params, settings['experiment_name'])

                right_params = optimized_params
                right_params[current_parameter] = right_params[current_parameter] * 1.5
                try: right_params['nrmse'] = n_experiments_chaotic(patterns, settings['n_rep'], **right_params)
                except Exception as e:
                    print(e)
                    break
                write_experiment_results(right_params, settings['experiment_name'])


                if min(left_params['nrmse'], right_params['nrmse']) < optimized_params['nrmse']:
                    if left_params['nrmse'] < right_params['nrmse']:
                        optimized_params = left_params
                    else:
                        optimized_params = right_params
                print(f"Optimizing parameter {current_parameter}, best value now is{optimized_params[current_parameter]}" +
                      f"after {counter} optimizing steps. Current NRMSE = {optimized_params['nrmse']}")


def write_experiment_results(results, filename):
    file_exists = os.path.isfile(filename)
    print(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()  # Write the header (keys)

        writer.writerow(results)  # Write the data (values)




