import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_nrmse_results_with_error_bars(folder_path, repetitions=30):
    """
    Plot NRMSE values with error bars from experiment results JSON files.

    Args:
        folder_path (str): Path to the folder containing the experiment results JSON files.
        repetitions (int): The specific number of repetitions to filter files for plotting.
    """
    # Dictionary to store data for each RFC type
    rfc_data = {}

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith('nrmse_results_') and file_name.endswith('.json'):
            parts = file_name.split('_')
            if len(parts) < 4:
                continue  # Skip malformed filenames

            # Extract M value, RFC type, and repetitions
            try:
                M_value = float(parts[2])
                rfc_type = parts[3]
                file_repetitions = int(parts[4].replace('.json', ''))
            except ValueError:
                continue  # Skip invalid filenames

            # Skip files with a different number of repetitions
            if file_repetitions != repetitions:
                continue

            # Read the JSON file
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Store data organized by RFC type
            if rfc_type not in rfc_data:
                rfc_data[rfc_type] = {}

            for M, nrmse_values in data.items():
                M = float(M)
                nrmse_values = np.array(nrmse_values, dtype=float)
                mean_nrmse = np.mean(nrmse_values)
                std_dev_nrmse = np.std(nrmse_values)


                if M not in rfc_data[rfc_type]:
                    rfc_data[rfc_type][M] = {'mean': [], 'std_dev': []}

                rfc_data[rfc_type][M]['mean'].append(mean_nrmse)
                rfc_data[rfc_type][M]['std_dev'].append(std_dev_nrmse)
    print(rfc_data)
    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 8))

    for rfc_type, M_values in rfc_data.items():
        M_sorted = sorted(M_values.keys())
        means = [np.mean(M_values[M]['mean']) for M in M_sorted]
        std_devs = [np.mean(M_values[M]['std_dev']) for M in M_sorted]
        if rfc_type == "matrix":
            ax.hlines(means[0], 100, 1000, label=f"matrix", colors = 'green')
        # Plot with error bars (2 standard deviations)
        ax.errorbar(
            M_sorted,
            means,
            yerr=[1 * sd for sd in std_devs],
            label=rfc_type,
            capsize=5,
            marker='o',
        )

    ax.set_title('NRMSE Results with Error Bars (1 sd)', fontsize=16)
    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('NRMSE', fontsize=14)
    ax.legend(title='RFC Type', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    folder_path = '../res/optimize_different_M_2'  # Replace with your folder path
    plot_nrmse_results_with_error_bars(folder_path, repetitions=30)
