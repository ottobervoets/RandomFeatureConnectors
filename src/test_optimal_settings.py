import numpy as np
import sys
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from plot_best import predict_choatic_systems
sys.path.append('/home3/s3417522/RandomFeatureConnectors')

# give a list of paths (which are "methods")

#from each path, extract all the data as pd dataframe
# then from each dataframe find the unique values for M
# for each unique value of M find the optimal settings and run the experiment rep times
# write each result to a dict + json


#from the dictionary, make the plot
# report mean and sample variance of these

def process_csv_files(path_to_folder):
    # Iterate through all files in the folder
    for file_name in os.listdir(path_to_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(path_to_folder, file_name)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the RFC type from the file (assumes a column or unique identifier named `rfc_type`)
            if 'rfc_type' not in df.columns:
                raise ValueError(f"The file {file_name} does not contain the column 'rfc_type'.")

            rfc_type = df['rfc_type'].iloc[0]  # Assumes all rows in the file have the same RFC type

            # Initialize a dictionary for this file
            optimal_parameters = {}

            # Group by the column M and find the row with the lowest nrmse for each unique M
            for m_value, group in df.groupby('M'):
                min_nrmse_row = group.loc[group['nrmse'].idxmin()].to_dict()
                optimal_parameters[m_value] = min_nrmse_row

            # Write the dictionary to a JSON file named after the RFC type
            output_file_name = f'optimal_parameters_per_M_{rfc_type}.json'
            output_file_path = os.path.join(path_to_folder, output_file_name)
            with open(output_file_path, 'w') as json_file:
                json.dump(optimal_parameters, json_file, indent=4)

    return f"Processed all files in {path_to_folder}."


def plot_nrmse_from_json(folder_path):
    # Initialize a dictionary to store data for plotting
    data_to_plot = {}

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('optimal_parameters_per_M_') and file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Extract the rfc_type from the file name
            rfc_type = file_name.split('optimal_parameters_per_M_')[1].split('.json')[0]

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract M and nrmse for plotting
            M_values = []
            nrmse_values = []
            for m_value, row_data in data.items():
                M_values.append(float(m_value))  # Convert M to float for proper sorting
                nrmse_values.append(row_data['nrmse'])

            # Store the data for plotting
            data_to_plot[rfc_type] = (sorted(M_values), [nrmse for _, nrmse in sorted(zip(M_values, nrmse_values))])

    # Plot the data
    plt.figure(figsize=(10, 6))
    for rfc_type, (M_values, nrmse_values) in data_to_plot.items():
        plt.plot(M_values, nrmse_values, label=f'RFC Type: {rfc_type}', marker='o')

    # Customize the plot
    plt.title('NRMSE vs. M for Different RFC Types')
    plt.xlabel('M')
    plt.ylabel('NRMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_individual_scatterplots_by_rfc(folder_path):
    # Initialize a dictionary to store the data
    M_colors = {}  # To ensure consistent coloring for the same M across files

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('optimal_parameters_per_M_') and file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract the RFC type from the file name
            rfc_type = file_name.replace('optimal_parameters_per_M_', '').replace('.json', '')

            # Extract data for numerical parameters
            data_to_plot = {}
            for m_value, row_data in data.items():
                m_value = float(m_value)  # Ensure M is treated as numerical
                if m_value not in M_colors:
                    M_colors[m_value] = plt.cm.tab10(len(M_colors))  # Assign a unique color

                for param, value in row_data.items():
                    # Filter out non-numerical, boolean, and invalid values
                    if (
                            param not in ['M', 'nrmse']
                            and isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and not np.ma.is_masked(value)
                            and not np.isnan(value)
                    ):
                        if param not in data_to_plot:
                            data_to_plot[param] = []
                        data_to_plot[param].append((m_value, value))

            # Remove parameters with all values equal
            filtered_data_to_plot = {}
            for param, values in data_to_plot.items():
                values_array = np.array([v[1] for v in values])
                if np.var(values_array) > 0:  # Check if the variance is non-zero
                    filtered_data_to_plot[param] = values

            # Scale the values for each parameter by dividing by their mean
            scaled_data_to_plot = {}
            for param, values in filtered_data_to_plot.items():
                values_array = np.array([v[1] for v in values])  # Extract the values
                mean_value = np.mean(values_array)
                if mean_value == 0:
                    mean_value = 1  # Avoid division by zero
                scaled_values = [(v[0], v[1] / mean_value) for v in values]  # Scale the values
                scaled_data_to_plot[f'{param} (mean={mean_value:.3e})'] = scaled_values

            # Create a scatter plot for this RFC type
            fig, ax = plt.subplots(figsize=(10, 6))
            for param, values in scaled_data_to_plot.items():
                for m_value, value in values:
                    ax.scatter(param, value, color=M_colors[m_value],
                               label=f'M={m_value}' if param == list(scaled_data_to_plot.keys())[0] else '')

            ax.set_title(f'Spread of Scaled Numerical Parameters (RFC Type: {rfc_type})')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Scaled Value')
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            ax.grid(True)

            # Show the plot
            plt.tight_layout()
            plt.show()


def perform_experiments(folder_path, n_rep, experiment_function, M_SETTING):
    """
    Perform experiments for each JSON file and save the results.

    Args:
        folder_path (str): Path to the folder containing the optimal parameters JSON files.
        n_rep (int): Number of repetitions for each experiment.
        experiment_function (callable): A function that simulates an experiment and returns an NRMSE value.
    """
    for file_name in os.listdir(folder_path):
        if file_name.startswith('optimal_parameters_per_M_') and file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Prepare results storage
            results = {}
            m_value = None
            # Perform experiments for each M
            for m_value, parameters in data.items():
                print(f"{m_value}, ")
                m_value = int(m_value)
                print("Mvalue = ", m_value)
                print(m_value, M_SETTING)
                if m_value != M_SETTING:
                    continue
                parameters['sample_rate'] = 0
                # Run experiments n_rep times and store the NRMSE values
                nrmse_values = []
                for i in range(n_rep):
                    print(parameters)
                    print("##########NREP", i)
                    return_dict = experiment_function(84, **parameters)
                    avg_nrmse = 0
                    for value in return_dict.values():
                        avg_nrmse += value['nrmse']
                    avg_nrmse /= len(return_dict)
                    nrmse_values.append(avg_nrmse)

                results[m_value] = nrmse_values
                break

            # Determine rfc type from the file name
            rfc_type = file_name.replace('optimal_parameters_per_M_', '').replace('.json', '')

            # Save results to a new JSON file
            results_file_name = f'nrmse_results_{m_value}_{rfc_type}_{n_rep}.json'
            results_file_path = os.path.join(folder_path, results_file_name)
            if len(results) > 0:
                with open(results_file_path, 'w') as results_file:
                    json.dump(results, results_file, indent=4)

                print(f"Results saved to: {results_file_path}")


# Example Experiment Function
def simulate_experiment(parameters):
    """
    Simulates an experiment and returns an NRMSE value.
    This is a placeholder function and should be replaced with your actual experiment logic.

    Args:
        parameters (dict): The parameters for the experiment.

    Returns:
        float: Simulated NRMSE value.
    """
    # Example: Generate a random NRMSE based on parameter values
    predict_choatic_systems(84, **parameters)
    return np.random.uniform(0, 1)  # Replace with your experiment logic


if __name__ == "__main__":

    path = "../res/matlab_maandag"
    process_csv_files(path)
    plot_nrmse_from_json(path)
    plot_individual_scatterplots_by_rfc(path)

    # arg_v = int(sys.argv[1])
    # arg_v = 8
    # M_settings = [100, 125, 187, 250, 312, 375, 500]#, 750, 1000]
    #750 en 1000 moeten nog.

    # for M in M_settings:
    # perform_experiments(folder_path=path, n_rep=30, experiment_function=predict_choatic_systems,
    #                     M_SETTING=187)
    # perform_experiments(folder_path=path, n_rep=30, experiment_function=predict_choatic_systems,
    #                     M_SETTING=250)

    # m_setting = M_settings[arg_v]
    # perform_experiments(folder_path=path, n_rep=30, experiment_function=predict_choatic_systems, M_SETTING=1)
