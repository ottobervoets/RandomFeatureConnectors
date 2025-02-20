import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_parameter_evolution_by_M(folder_path, exclude_params=None):
    """
    Create line plots for numerical parameters' evolution across different M values.
    Each RFC type gets its own plot, showing how each parameter evolves as M changes.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        exclude_params (list, optional): List of parameter names to exclude from the plots. Default is None.
    """
    if exclude_params is None:
        exclude_params = []

    # Dictionary to assign consistent colors to M values
    M_colors = {}

    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.startswith('optimal_parameters_per_M_') and file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Load the JSON data
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract the RFC type from the file name
            rfc_type = file_name.replace('optimal_parameters_per_M_', '').replace('.json', '')

            # Prepare data for plotting
            parameter_data = {}
            for m_value, row_data in data.items():
                m_value = float(m_value)  # Ensure M is numerical
                if m_value not in M_colors:
                    M_colors[m_value] = plt.cm.tab10(len(M_colors))  # Assign a unique color

                for param, value in row_data.items():
                    # Check if the value is a valid numerical parameter and not excluded
                    if (
                        param not in exclude_params
                        and param not in ['M', 'nrmse']
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and not np.ma.is_masked(value)
                        and not np.isnan(value)
                    ):
                        if param not in parameter_data:
                            parameter_data[param] = []
                        parameter_data[param].append((m_value, value))

            # Filter out parameters with all equal values
            filtered_data = {
                param: values
                for param, values in parameter_data.items()
                if np.var([v[1] for v in values]) > 0
            }

            # Scale the parameter values by their mean
            scaled_data = {}
            for param, values in filtered_data.items():
                m_values, param_values = zip(*values)
                mean_value = np.mean(param_values)
                if mean_value == 0:
                    mean_value = 1  # Avoid division by zero
                scaled_values = [(m, val / mean_value) for m, val in values]
                scaled_values = [(m, val / 1) for m, val in values]
                scaled_data[f'{param} (mean={mean_value:.3e})'] = scaled_values

            # Create the line plot for this RFC type
            fig, ax = plt.subplots(figsize=(12, 8))
            for param, values in scaled_data.items():
                m_values, scaled_values = zip(*values)
                m_values, scaled_values = zip(*sorted(zip(m_values, scaled_values)))  # Sort by M
                ax.plot(
                    m_values,
                    scaled_values,
                    marker='o',
                    label=param
                )

            # Customize plot
            ax.set_title(f'Parameter Evolution Across M (RFC Type: {rfc_type})')
            ax.set_xlabel('M Value')
            ax.set_ylabel('Scaled Parameter Values')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)

            # Show the plot
            plt.tight_layout()
            plt.show()

# Example usage

excluded_parameters = [
    'beta_W_out',
    'beta_G',
    'beta_d',
    'aperture_rossler_attractor_2d',
    'aperture_lorenz_attractor_2d',
    'aperture_mackey_glass_2d',
    'aperture_henon_attractor_2d',
    'spectral_radius',
    'noise_std',
    'signal_noise',
    'max_n_features',
    'W_in_std',
    'W_sr',
    'beta_D',
    'b_std'
]


# Example usage
folder_path = ("../../res/matlab_woensdag")  # Replace with your folder path
plot_parameter_evolution_by_M(folder_path, exclude_params=excluded_parameters)


