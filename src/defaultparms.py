from datetime import datetime

default_parms = {
    'n_harvest': 400,
    'washout': 500,
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
    'F_method': "patterns",
    'signal_dim': 2,
    'G_method': "W_F",
    'noise_std': None,
    'rfc_type': 'PCARFC',
    'max_n_features': 100,
    'verbose': True
}

default_parmas_chaotic = {
    'n_harvest': 2000,
    'washout': 500,
    'beta_W_out': 5,
    'beta_G': 0.1,
    'beta_D': 0.07,
    'aperture_rossler_attractor': 7000,
    'aperture_lorenz_attractor': 1000,
    'aperture_mackey_glass': 150,
    'aperture_henon_attractor': 20,
    'spectral_radius': 1.4,
    'N': 500,
    'M': 1000,
    'n_adapt': 2000,
    'W_sr': 0.5,
    'W_sparseness': 0.1,
    'd_dim': "reservoir_dim",
    'F_method': "patterns",
    'signal_dim': 2,
    'G_method': "W_F",
    'noise_mean': None,
    'noise_std': 0.001,
    'rfc_type': 'PCARFC',
    'max_n_features': 750,
    'verbose': False

}

parameters_to_optimize = {  # idea for optmization.
    'noise_std': {
        'step_type': 'relative',
        'step_size': 0.50,
        'boundaries': [0, 0.5]
    },
    'beta_G': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [0, 10]
    },
    'aperture_rossler_attractor': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    'aperture_lorenz_attractor': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    'aperture_mackey_glass': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    'aperture_henon_attractor': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    'beta_W_out': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [0, 10]
    },
    'beta_D': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [0, 10]
    },
    'spectral_radius': {
        'step_type': 'absolute',
        'step_size': 0.1,
        'boundaries': [0, 2]
    },
}


optimization_settings = {
    'experiment_name': "../res/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv",
    'cycles': 3,
    'n_rep': 2
}
'''
    'beta_W_out': 0.01,
    'beta_G': 1,
    'beta_D': 0.01,
    'aperture': 8,
'''
# parameters_to_optimzie = ["beta_G",
#                           "aperture",
#                           "beta_W_out",
#                           "max_n_features",
#                           "beta_D"]
