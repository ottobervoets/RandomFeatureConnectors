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
    # 'aperture_rossler_attractor': 48,
    # 'aperture_lorenz_attractor': 28,
    # 'aperture_mackey_glass': 30,
    # 'aperture_henon_attractor': 18,
    'aperture_rossler_attractor_2d': 48,# 10 ** 2.8,  # 48,
    'aperture_lorenz_attractor_2d': 28,#10 ** 2.6,  # 15,
    'aperture_mackey_glass_2d': 30,#10 ** 3.1,  # 35,
    'aperture_henon_attractor_2d': 18,#10 ** 3,  # 18,
    'spectral_radius': 1.1,
    'N': 250,
    'M': 750,
    'n_adapt': 3000,
    'W_sparseness': 0.1,
    'd_dim': "reservoir_dim",
    'F_method': "patterns",
    'signal_dim': 2,
    'G_method': "W_F",
    'noise_mean': 0.00001,
    'noise_std': 0.0005,
    'signal_noise' : 0.007,
    'rfc_type': 'PCARFC',
    'max_n_features': 60,
    'verbose': False,
    'W_in_std': 1.5,
    'W_sr': 1.2,
    'sample_rate': 15,
    'verbose': False,
    'b_std': 0.4
    # 'b_in_std'
}

default_parmas_matrix = {
    'N': 500,
    'M': 1,
    'rfc_type': 'matrix_conceptor', #10 ** 3, 10 ** 2.6, 10 ** 3.1, 10 ** 2.8
    'aperture_rossler_attractor': 10**2.8, #48,
    'aperture_lorenz_attractor':  10**2.6,#15,
    'aperture_mackey_glass': 10**3.1, #35,
    'aperture_henon_attractor':  10**3,#18,
    'aperture_rossler_attractor_2d': 10**2.8, #48,
    'aperture_lorenz_attractor_2d':  10**2.6,#15,
    'aperture_mackey_glass_2d': 10**3.1, #35,
    'aperture_henon_attractor_2d':  10**3,#18,
    'n_adapt': 3000,
    'washout': 500,
    'beta_W': 0.0001,
    'beta_W_out': 0.01,
    'W_in_std': 1.2,
    'W_sr': 0.6,
    'bias': 0.4,
    'noise_std': 0.0001,
    'signal_noise': 0.0005,
}

parameters_to_optimize_matrix = {  # idea for optmization.
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
    'beta_W': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [0, 10]
    },
    # 'aperture_rossler_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.25,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_lorenz_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.25,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_mackey_glass': {
    #     'step_type': 'relative',
    #     'step_size': 0.25,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_henon_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.1,
    #     'boundaries': [1, 10_000],
    # },

    'aperture_rossler_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_lorenz_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_mackey_glass_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_henon_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    'beta_W_out': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [0, 10]
    },
    'W_in_std': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0, 2]
    },
    'W_sr': {
        'step_type': 'absolute',
        'step_size': 0.1,
        'boundaries': [0, 2]
    },
    'bias': {
        'step_type': 'absolute',
        'step_size': 0.1,
        'boundaries': [0, 2]

    }
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

    'max_n_features': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [0, 10]
    },
    'aperture_rossler_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_lorenz_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_mackey_glass_2d': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [1, 10_000],
    },
    'aperture_henon_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [1, 10_000],
    },
    # 'aperture_rossler_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.1,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_lorenz_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.1,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_mackey_glass': {
    #     'step_type': 'relative',
    #     'step_size': 0.1,
    #     'boundaries': [1, 10_000],
    # },
    # 'aperture_henon_attractor': {
    #     'step_type': 'relative',
    #     'step_size': 0.1,
    #     'boundaries': [1, 10_000],
    # },
    'beta_W_out': {
        'step_type': 'relative',
        'step_size': 0.25,
        'boundaries': [0, 10]
    },
    'beta_D': {
        'step_type': 'relative',
        'step_size': 0.1,
        'boundaries': [0, 10]
    },
    'spectral_radius': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]
    },
    'W_in_std': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]
    },
    'W_sr': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]
    },
    'b_std': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]
    }
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
