from datetime import datetime

default_parms = {
    'n_harvest': 400,
    'washout': 500,
    'beta_W_out': 0.01,
    'beta_G': 1,
    'beta_D': 0.01,
    'aperture': 8,
    'spectral_radius': 1.4,
    'N': 500,
    'M': 2500,
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
    'n_harvest': 3000,
    'washout': 500,
    'beta_W_out': 0.02,
    'beta_G': 0.4,
    'beta_D': (6*(10**-5)),
    'aperture_rossler_attractor': 48,
    'aperture_lorenz_attractor': 28,
    'aperture_mackey_glass': 130,
    'aperture_henon_attractor': 18,
    'spectral_radius': 1.1,
    'N': 300,
    'M': 1200,
    'n_adapt': 3000,
    'W_sparseness': 0.1,
    'd_dim': "reservoir_dim",
    'F_method': "patterns",
    'signal_dim': 2,
    'G_method': "W_F",
    'noise_mean': None,
    'noise_std': 0.0005,
    'signal_noise' : 0.007,
    'rfc_type': 'PCARFC',
    'max_n_features': 750,
    'verbose': False,
    'W_in_std': 1.5,
    'W_sr': 1.2

}

parameters_to_optimize = {  # idea for optmization.
    'noise_std': {
        'step_type': 'relative',
        'step_size': 0.10,
        'boundaries': [0, 0.5]
    },
    'signal_noise': {
        'step_type': 'relative',
        'step_size': 0.10,
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
    'W_in_std': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0, 2]
    },
    'W_sr': {
        'step_type': 'absolute',
        'step_size': 0.05,
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
