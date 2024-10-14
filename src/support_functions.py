import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def compute_nrmse(true_series, predicted_series, phase_shift=0):
    # Shift the predicted series by phase_shift steps
    aligned_predicted = predicted_series[phase_shift:]
    aligned_true = true_series

    # Compute NRMSE
    min_len = min(len(true_series), len(predicted_series))
    p = np.array(aligned_predicted[0:min_len])
    y = aligned_true[0:min_len]
    nrmse = np.sqrt(np.mean((y-p)**2)/np.mean(p**2))
    return nrmse


def find_optimal_phase_shift(true_series, predicted_series, max_shift):
    best_nrmse = float('inf')
    best_shift = 0


    for shift in range(max_shift + 1):
        nrmse = compute_nrmse(true_series, predicted_series, phase_shift=shift)
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_shift = shift
            # print(best_shift, best_nrmse)

    return best_shift, best_nrmse


def plot_aligned_series_with_optimal_shift(true_series, predicted_series, max_shift=20, segment_range=None):
    """
    Plot a segment of both time series in the same figure, optimally phase-aligned, and show NRMSE.

    Args:
        true_series (list or np.array): The true time series data.
        predicted_series (list or np.array): The predicted time series data.
        max_shift (int): Maximum phase shift to test.
        segment_range (tuple): A tuple specifying the range of time steps to plot (start, end).

    Returns:
        None: Displays the plot.
    """
    # Find the optimal phase shift
    optimal_shift, best_nrmse = find_optimal_phase_shift(true_series, predicted_series, max_shift)

    # Apply the optimal phase shift for plotting
    aligned_predicted = predicted_series[optimal_shift:optimal_shift+len(true_series)]
    aligned_true = true_series
    # Plot the two time series
    plt.figure(figsize=(10, 6))
    plt.plot(aligned_true, label="True Series", color='b')
    plt.plot(aligned_predicted, label="Predicted Series", color='r', linestyle='--')

    # Display NRMSE and optimal shift in the title
    plt.title(f"Optimal Phase Shift = {optimal_shift}, NRMSE = {best_nrmse:.4f}")
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show(block=False)


# Example usage:
# true_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# predicted_series = [1.1, 1.9, 3.2, 4.1, 5.1, 5.9, 6.8, 8.0, 9.1, 9.9]
#
# # Call function to compute NRMSE and plot the time series with the optimal phase shift
# plot_aligned_series_with_optimal_shift(true_series, predicted_series, max_shift=20, segment_range=(0, 10))


def plot_internal(internal_values, neurons, time, plot_time = None):
    for neuron in neurons:
        plt.plot(internal_values[neuron][time], label=f"{neuron}")
    plt.legend()
    if plot_time:
        plt.show(block=False)
        plt.pause(plot_time)
        plt.close("all")
        return
    plt.show()

def transpose_internal(recordings):
    return list(map(list, zip(*recordings)))


import os
from datetime import datetime
import matplotlib.pyplot as plt


def create_experiment_dir(save_string,
                          figures,
                          base_dir = "/res/experiments/"):
    # Step 1: Create the directory with current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, current_time)

    # Create the directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Step 2: Write the provided string to a text file
    file_path = os.path.join(experiment_dir, "experiment_info.txt")
    with open(file_path, "w") as f:
        f.write(save_string)

    # Step 3: Save the figures in the directory
    for i, fig in enumerate(figures):
        figure_path = os.path.join(experiment_dir, f"figure_{i + 1}.png")
        fig.savefig(figure_path)

    print(f"Experiment directory created at: {experiment_dir}")


