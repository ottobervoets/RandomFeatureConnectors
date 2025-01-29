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
    "n_harvest": 3000,
    "washout": 500,
    "beta_W_out": 0.02,
    "beta_G": 0.4356,
    "beta_D": 6.000000000000001e-05,
    "aperture_rossler_attractor_2d": 48.0,
    "aperture_lorenz_attractor_2d": 28.0,
    "aperture_mackey_glass_2d": 30.0,
    "aperture_henon_attractor_2d": 18.0,
    "spectral_radius": 1.1,
    "N": 250,
    "M": 1000,
    "n_adapt": 3000,
    "W_sparseness": 0.1,
    "d_dim": "reservoir_dim",
    "F_method": "patterns",
    "signal_dim": 2,
    "G_method": "W_F",
    "noise_mean": 0,
    "noise_std": 0.0003644999999999,
    "signal_noise": 0.00693,
    "rfc_type": "PCARFC",
    "max_n_features": 275.0,
    "verbose": False,
    "W_in_std": 1.5,
    "W_sr": 1.2,
    "sample_rate": 15,
    "b_std": 0.4,
    "experiment_name": "../res/RFC_MAANDAG__250.csv",
    "cycles": 3,
    "n_rep": 5,
    "nrmse": 0.002112948322925
}

default_parmas_matrix_500 = {
    "N": 500,
    "M": 500,
    "rfc_type": "matrix_conceptor",
    # "aperture_rossler_attractor": 10**3, #50.0,
    # "aperture_lorenz_attractor": 10**2.6,#20.0,
    # "aperture_mackey_glass": 10**3.1,#35.0,
    # "aperture_henon_attractor": 10**2.8,#20.0,
    'aperture_rossler_attractor_2d': 10 ** 2.8,  # 48,
    'aperture_lorenz_attractor_2d':  10 ** 2.6,  # 15,
    'aperture_mackey_glass_2d': 10**3.1,# 10 ** 3.1,  # 35,
    'aperture_henon_attractor_2d': 10 ** 3,  # 18,
    "n_adapt": 1900,
    "washout": 500,
    "beta_W": 0.0001,
    "beta_W_out": 0.01,
    "W_in_std": 1.2,
    "W_sr": 0.6,
    "noise_std":None,# 0.00011,
    "signal_noise":None,# 0.0005,
    "bias": 0.4
}

best_matrix_parms_250 = {
    "N": 250,
    "M": 500,
    "rfc_type": "matrix_conceptor",
    "aperture_rossler_attractor_2d": 50.0,
    "aperture_lorenz_attractor_2d": 20.0,
    "aperture_mackey_glass_2d": 35.0,
    "aperture_henon_attractor_2d": 20.0,
    "n_adapt": 3000,
    "washout": 500,
    "beta_W": 0.0001,
    "beta_W_out": 0.01,
    "W_in_std": 1.2,
    "W_sr": 0.8,
    "noise_std": 0.00011,
    "signal_noise": 0.0005,
    'bias': 0.4
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
        'step_size': 0.2,
        'boundaries': [1, 10_000],
    },
    'aperture_lorenz_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.2,
        'boundaries': [1, 10_000],
    },
    'aperture_mackey_glass_2d': {
        'step_type': 'relative',
        'step_size': 0.2,
        'boundaries': [1, 10_000],
    },
    'aperture_henon_attractor_2d': {
        'step_type': 'relative',
        'step_size': 0.2,
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
        'boundaries': [.1, 2]
    },
    'W_sr': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]
    },
    'bias': {
        'step_type': 'absolute',
        'step_size': 0.05,
        'boundaries': [0.1, 2]

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
        'step_size': 0.25,
        'boundaries': [0, 1000]
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
        'step_size': 0.25,
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
        'step_size': 0.25,
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
