import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def compute_nrmse(true_series, predicted_series, phase_shift=0):
    """
    Compute the NRMSE between the true and predicted series after applying phase alignment.
    Args:
        true_series (list or np.array): The true time series data.
        predicted_series (list or np.array): The predicted time series data.
        phase_shift (int): Number of steps to shift the predicted series for alignment.

    Returns:
        float: NRMSE value between the aligned time series.
    """
    # Shift the predicted series by phase_shift steps
    if phase_shift > 0:
        aligned_predicted = predicted_series[phase_shift:]
        aligned_true = true_series[:len(aligned_predicted)]
    else:
        aligned_predicted = predicted_series[:len(predicted_series) + phase_shift]
        aligned_true = true_series[-phase_shift:]

    # Compute NRMSE
    min_len = min(len(true_series), len(predicted_series))
    p = np.array(aligned_predicted[0:min_len])
    y = aligned_true[0:min_len]
    nrmse = np.sqrt(np.mean((y-p)**2)/np.mean(p**2))
    # print(nrmse)
    return nrmse


def find_optimal_phase_shift(true_series, predicted_series, max_shift):
    """
    Find the optimal phase shift that minimizes the NRMSE.

    Args:
        true_series (list or np.array): The true time series data.
        predicted_series (list or np.array): The predicted time series data.
        max_shift (int): Maximum phase shift to test.

    Returns:
        int: Optimal phase shift that minimizes NRMSE.
        float: Corresponding NRMSE.
    """

    best_nrmse = float('inf')
    best_shift = 0


    for shift in range(max_shift + 1):
        nrmse = compute_nrmse(true_series, predicted_series, phase_shift=shift)
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_shift = shift

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
    if optimal_shift > 0:
        aligned_predicted = predicted_series[optimal_shift:]
        aligned_true = true_series[:len(aligned_predicted)]
    else:
        aligned_predicted = predicted_series[:len(predicted_series) + optimal_shift]
        aligned_true = true_series[-optimal_shift:]

    # Limit to the specified segment range
    if segment_range is not None:
        start, end = segment_range
        aligned_predicted = aligned_predicted[start:end]
        aligned_true = aligned_true[start:end]

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
