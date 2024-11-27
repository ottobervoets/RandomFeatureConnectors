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
    'n_harvest': 400,
    'washout': 500,
    'beta_W_out': 0.01,
    'beta_G': 1,
    'beta_D': 0.01,
    'aperture': 200,
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
    'noise_mean': None,
    'noise_std': 0.001,
    'rfc_type': 'PCARFC',
    'max_n_features': 500,
    'verbose': False

}

parameters_to_optimize = {  # idea for optmization.
    'beta_G': {
        'step_type': 'relative',
        'step_size': 0.10,
        'boundaries': [0, 10]
    },
    'aperture': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 1000],
    },
    'beta_W_out': {
        'step_type': 'relative',
        'step_size': 0.10,
        'boundaries': [0, 10]
    },
    'beta_D': {
        'step_type': 'relative',
        'step_size': 0.10,
        'boundaries': [0, 10]
    },
    'max_n_features': {
        'step_type': 'absolute',
        'step_size': 20,
        'boundaries': [0,150]
    },
    'noise_std': {
        'step_type': 'relative',
        'step_size': 0.50,
        'boundaries': [0, 0.5]
    }
}


optimization_settings = {
    'experiment_name': "../res/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv",
    'cycles': 2,
    'n_rep': 5
}
'''
    'beta_W_out': 0.01,
    'beta_G': 1,
    'beta_D': 0.01,
    'aperture': 8,
'''
parameters_to_optimzie = ["beta_G",
                          "aperture",
                          "beta_W_out",
                          "max_n_features",
                          "beta_D"]
