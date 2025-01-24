import matplotlib.pyplot as plt
import os
import json
import numpy as np
import seaborn as sns

def plot_nrmse_results_with_measurements_same_color(folder_path, repetitions=30):
    """
    Plot NRMSE values with individual measurements as small points and means as lines,
    ensuring all elements related to the same RFC type are the same color.

    Args:
        folder_path (str): Path to the folder containing the NRMSE results JSON files.
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
                if M not in rfc_data[rfc_type]:
                    rfc_data[rfc_type][M] = []
                rfc_data[rfc_type][M].extend(nrmse_values)

    # Generate a consistent color palette
    palette = sns.color_palette("tab10", n_colors=len(rfc_data))
    rfc_colors = {rfc: palette[i] for i, rfc in enumerate(rfc_data.keys())}

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 8))

    for rfc_type, M_values in rfc_data.items():
        M_sorted = sorted(M_values.keys())
        means = [np.mean(M_values[M]) for M in M_sorted]

        # Plot mean line
        ax.plot(
            M_sorted,
            means,
            label=rfc_type,
            marker='o',
            linewidth=1,
            markersize=2,
            color=rfc_colors[rfc_type]
        )

        # Plot individual points
        for M in M_sorted:
            ax.scatter(
                [M] * len(M_values[M]),
                M_values[M],
                alpha=0.8,
                s=2,
                color=rfc_colors[rfc_type],
                label=None  # Avoid duplicate legend entries for points
            )

    # ax.set_title('NRMSE Results with Individual Measurements', fontsize=16)
    ax.set_xlabel('M Values', fontsize=14)
    ax.set_ylabel('NRMSE', fontsize=14)
    ax.legend(title='RFC Type', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage


folder_path = "../res/optimize_different_M_2"  # Replace with your folder path
plot_nrmse_results_with_measurements_same_color(folder_path, repetitions=30)



