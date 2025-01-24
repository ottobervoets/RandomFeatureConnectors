import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os
import json
import numpy as np
import seaborn as sns

def plot_binned_nrmse_results_with_overlap_and_spacing(
    folder_path, repetitions=30, grid_shape=(3, 4), num_bins=20, bar_spacing=0.05
):
    """
    Create a grid of bar plots for each M value, showing the binned NRMSE counts for all RFC types,
    with overlapping, transparent bars and controllable spacing.

    Args:
        folder_path (str): Path to the folder containing the NRMSE results JSON files.
        repetitions (int): The specific number of repetitions to filter files for plotting.
        grid_shape (tuple): Shape of the subplot grid (rows, cols).
        num_bins (int): Number of bins to divide the NRMSE values into.
        bar_spacing (float): Spacing between bars for visual clarity (0 means bars touch completely).
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

    # Determine all unique M values
    unique_M_values = sorted(
        set(M for rfc in rfc_data.values() for M in rfc.keys())
    )

    # Generate a consistent color palette
    palette = sns.color_palette("tab10", n_colors=len(rfc_data))
    rfc_colors = {rfc: palette[i] for i, rfc in enumerate(rfc_data.keys())}

    # Determine the number of subplots
    num_plots = len(unique_M_values)
    rows, cols = grid_shape
    if rows * cols < num_plots:
        raise ValueError("Grid shape is too small to accommodate all M values.")

    # Create the figure and axes for the grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharey=True)
    axes = axes.flatten()  # Flatten to iterate easily

    for i, M in enumerate(unique_M_values):
        ax = axes[i]

        # Bin data for each RFC type
        max_nrmse = max(max(M_values.get(M, [0])) for M_values in rfc_data.values())
        min_nrmse = min(min(M_values.get(M, [0])) for M_values in rfc_data.values())
        bins = np.linspace(min_nrmse, max_nrmse, num_bins + 1)

        bin_width = (bins[1] - bins[0]) * (1 - bar_spacing)  # Adjust width for spacing
        offset = (bins[1] - bins[0]) / len(rfc_data)  # Offset to align bars properly
        for j, (rfc_type, M_values) in enumerate(rfc_data.items()):

            if M not in M_values:
                continue

            # Count occurrences within bins
            counts, _ = np.histogram(M_values[M], bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoints of bins

            # Plot bars with proper alignment and spacing
            ax.bar(
                bin_centers,# + (j - len(rfc_data) / 2) * offset,  # Adjust offset for alignment
                counts,
                width=bin_width,  # Adjusted width for spacing
                color=rfc_colors[rfc_type],
                alpha=0.5,  # Transparency for overlap
                label=rfc_type if i == 0 else None
            )
        # print(rfc_data)
        _, t_test = ttest_ind(rfc_data['PCARFC'][M], rfc_data['base'][M], alternative='less', equal_var=False)
        # Customize subplot
        ax.set_title(f'M = {M:.0f}, {t_test:.6f}', fontsize=10)
        ax.set_xlabel('NRMSE', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Adjust layout and add a legend
    fig.suptitle('average NRMSE per M', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    fig.legend(
        title='RFC Type',
        handles=[plt.Line2D([0], [0], color=rfc_colors[rfc], label=rfc) for rfc in rfc_data.keys()],
        loc='upper right',
        # bbox_to_anchor=(0.5, 0.02),
        ncol=len(rfc_data),
        fontsize=10
    )

    plt.show()


folder_path = '../../res/optimize_different_M_2'  # Replace with your folder path
plot_binned_nrmse_results_with_overlap_and_spacing(
    folder_path, repetitions=30, grid_shape=(3, 3), num_bins=30, bar_spacing=0
)


