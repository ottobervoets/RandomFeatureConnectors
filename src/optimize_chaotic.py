import sys
sys.path.append('/home3/s3417522/RandomFeatureConnectors')
from src.support_functions import *

import numpy as np
import csv
from defaultparms import *
from datetime import datetime
from models.factory import create_RFC
from signal_generators import rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor
from matlab_copy.helper_functions import (rossler_attractor_2d,
                                          mackey_glass_2d,
                                          lorenz_attractor_2d,
                                          henon_attractor_2d)


def one_experiment_chaotic(test_pattern: dict, rfc_type: str, test_length: int = 84, **kwargs):
    '''

    :param test_pattern:
    :param rfc_type:
    :param test_length:
    :param kwargs:
    :return:
    '''

    rfc = create_RFC(rfc_type, **kwargs)
    rfc.store_patterns(**kwargs)
    nrmses = []
    for key, value in test_pattern.items():
        result = rfc.record_chaotic(length=test_length, pattern_name=key)
        nmrse = NRMSE_2_dim(test_pattern[key],
                            result)
        # if nmrse > 0.5:
            # print(f"NMRSE large{nmrse} for pattern {test_pattern.keys()}")
        nrmses.append(nmrse)

    return np.mean(nrmses), nrmses


def n_experiments_chaotic(n_rep, rfc_type, test_length = 84, **kwargs):
    print(f"do experiment repeat {n_rep} times")
    results = []

    rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor = rossler_attractor_2d, lorenz_attractor_2d, mackey_glass_2d, henon_attractor_2d
    #
    pattern_generators = [rossler_attractor, lorenz_attractor, mackey_glass, henon_attractor]
    # pattern_generators = [rossler_attractor_2d,  #these are those from herbert.
    #                       lorenz_attractor_2d,
    #                       mackey_glass_2d,
    #                       henon_attractor_2d]
    # pattern_generators = [rossler_attractor, mackey_glass, henon_attractor]
    # add training pattern to params.
    kwargs['training_patterns'] = {}
    kwargs['apperture'] = {}
    true_patterns = {}  # todo make dict, just like the starting values.
    training_length = kwargs['n_adapt'] + kwargs['washout']
    total_pattern = {}
    for pattern_generator in pattern_generators:
        total_pattern[pattern_generator.__name__] = pattern_generator(total_time=training_length + test_length)
        kwargs['training_patterns'][pattern_generator.__name__] = total_pattern[pattern_generator.__name__][
                                                                       0:training_length]
        true_patterns[pattern_generator.__name__] = total_pattern[pattern_generator.__name__][
                                                    training_length:(training_length + test_length)]

    for experiment in range(n_rep):
        mean_nrmses = float('Nan')
        while(np.isnan(mean_nrmses)):
            mean_nrmses, _ = one_experiment_chaotic(test_pattern=true_patterns,
                                                    rfc_type=rfc_type, test_length=test_length,
                                                    **kwargs)
        results.append(mean_nrmses)
    return np.mean(results)


def parameter_step(current_parameter_value, info_dict, direction):
    # Determine the step value first
    match info_dict["step_type"]:
        case "relative":
            step = current_parameter_value * info_dict["step_size"]
        case "absolute":
            step = info_dict["step_size"]
        case _:
            raise ValueError("Invalid step_type provided")

    # Determine how to apply the step based on direction
    match direction:
        case "right":
            return current_parameter_value + step
        case "left":
            return current_parameter_value - step
        case _:
            raise ValueError("Invalid direction provided")
def do_and_write_experiment(settings, **params):
    params['nrmse'] = n_experiments_chaotic(**params)
    # try:
    #     params['nrmse'] = n_experiments_chaotic(**params)
    # except Exception as e:
    #     print(e)
    #     return
    write_experiment_results(params, settings['experiment_name'])
    return params

def go_one_driection(current_param, info_dict, params, direction, settings):
    while True:
        print(f"taking a step {direction}, for parameter {current_param} at value {params[current_param]}, nrmse = {params['nrmse']}")
        params[current_param] = parameter_step(params[current_param], info_dict, direction)
        if params[current_param] < info_dict['boundaries'][0] or params[current_param] > info_dict['boundaries'][1]:
            print(f"Parameter {current_param} is over the boundary with value {params[current_param]}")
            break
        temp = do_and_write_experiment(settings, **params)
        if temp['nrmse'] > params['nrmse']:
            break
        params = temp
    return params



def optimize_parameters_chaotic(parameters_to_optimize, default_parms, optimization_settings):
    # get starting parameters
    optimized_params = default_parms.copy()

    default_settings = {
        'experiment_name': "../res/" + "RFC_VRIJDAG_" + f"_{optimized_params['N']}" + ".csv",
        'cycles': 3,
        'n_rep': 5
    }
    optimized_params = {**optimized_params, **default_settings}
    settings = default_settings
    print(f"Optimizing with the following settings {settings}")
    for _ in range(settings['cycles']):
        for current_parameter, info_dict in parameters_to_optimize.items():
            print("##################", current_parameter)
            optimized_params['nrmse'] = n_experiments_chaotic(**optimized_params)
            print(f"NRSME {optimized_params['nrmse']}, for parameter {current_parameter} "
                  f"at value {optimized_params[current_parameter]}")
            left_params = optimized_params.copy()
            right_params = optimized_params.copy()
            left_params['nrmse'] = 0
            right_params['nrmse'] = 0
            left_params[current_parameter] = parameter_step(current_parameter_value=optimized_params[current_parameter],
                           info_dict=info_dict,
                           direction='left')

            right_params[current_parameter] = parameter_step(current_parameter_value=optimized_params[current_parameter],
                                          info_dict=info_dict,
                                          direction='right')

            right_params = do_and_write_experiment(settings, **right_params)
            left_params = do_and_write_experiment(settings, **left_params)
            direction = None
            if left_params['nrmse'] < optimized_params['nrmse']:
                direction = "left"
            elif right_params['nrmse'] < optimized_params['nrmse']:
                direction = "right"
                # go left
            else:
                print(f"No direction, parameter {current_parameter} is already optimal at value "
                      f"{optimized_params[current_parameter]} and nrmse = {optimized_params['nrmse']}")
                continue
            optimized_params = go_one_driection(params=left_params,
                             info_dict=info_dict,
                             current_param=current_parameter,
                             direction=direction,
                             settings=settings)
            print(
                f"Otimized parameter {current_parameter}, best value now is{optimized_params[current_parameter]}" +
                f"current NRMSE = {optimized_params['nrmse']}")


def write_experiment_results(results, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()  # Write the header (keys)

        writer.writerow(results)  # Write the data (values)


if __name__ == "__main__":
    # arg_v = sys.argv[1]
    arg_v = 3
    M_settings = [100, 125, 187, 250, 312, 375, 500, 750, 1000]
    default_parmas_chaotic['M'] = M_settings[arg_v]
    default_parmas_chaotic['N'] = 250
    default_parmas_chaotic['rfc_type'] = 'PCARFC'
    default_parmas_chaotic['max_n_features'] = np.min([default_parmas_chaotic['N'], default_parmas_chaotic['M']/4])
    print(parameters_to_optimize.keys())
    optimize_parameters_chaotic(parameters_to_optimize=parameters_to_optimize, default_parms=default_parmas_chaotic,
                                optimization_settings=optimization_settings)

    # default_parmas_chaotic['rfc_type'] = 'base_RFC'
    # optimize_parameters_chaotic(optimization_settings=optimization_settings, **default_parmas_chaotic)
