from models.matrix_conceptor import MatrixConceptor
from src.signal_generators import rossler_attractor, henon_attractor, lorenz_attractor, mackey_glass

import matplotlib.pyplot as plt


if __name__ == "__main__":

    signal_generators = [rossler_attractor, henon_attractor, lorenz_attractor, mackey_glass]
    matrix_conceptor = MatrixConceptor()
    training_length = 3500
    prediction_length = 84
    patterns = {}
    true_values = {}
    apertures = {
        'rossler_attractor': 500,
        'henon_attractor': 630,
        'lorenz_attractor': 400,
        'mackey_glass': 1300

    }
    # apertures = {
    #     'rossler_attractor': 30,
    #     'henon_attractor': 30,
    #     'lorenz_attractor': 40,
    #     'mackey_glass': 30
    # }

    for signal_generators in signal_generators:
        complete_sequence = signal_generators(total_time = training_length+prediction_length)
        patterns[signal_generators.__name__] = complete_sequence[:training_length]
        true_values[signal_generators.__name__] = complete_sequence[training_length:]
    matrix_conceptor.store_patterns(training_patterns=patterns, apertures=apertures)

    predictions = {}
    for name  in patterns.keys():
        predictions[name] = matrix_conceptor.predict_n_steps(prediction_length, name)

    predicted_values = predictions

    # Determine the number of plots and create a 2x2 grid
    num_functions = len(true_values)
    rows = cols = 2  # Start with a 2x2 grid

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = axes.flatten()  # Flatten to easily iterate over subplots

    # Iterate over the dictionary keys and plot
    for i, key in enumerate(true_values.keys()):
        if i >= len(axes):
            break  # Stop if there are more functions than subplots
        ax = axes[i]

        # Extract data
        true_data = true_values[key]

        predicted_data = predicted_values[key]
        # Separate x and y values
        true_x, true_y = zip(*true_data)
        pred_x, pred_y = zip(*predicted_data)

        # Plot true and predicted values
        ax.plot(true_x, true_y, label="True", marker='o', linewidth = 1, markersize=1)
        ax.plot(pred_x, pred_y, label="Predicted", marker='o', linewidth = 1, markersize=1)
        ax.plot(true_x[0], true_y[0], color= 'green', marker='o', markersize=5)
        ax.plot(pred_x[0], pred_y[0], color='red', marker='o', markersize=5)
        ax.set_title(key)
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if there are fewer functions than subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


