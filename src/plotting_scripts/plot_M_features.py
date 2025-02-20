import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data from file
json_path = "../../res/matlab_woensdag/optimal_parameters_per_M_PCARFC.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Extract values for each experiment
experiments = list(data.keys())  # Experiment names
M_values = [data[exp]["M"] for exp in experiments]  # Extract M for each experiment
max_n_features_values = [data[exp]["max_n_features"] * 4 for exp in experiments]  # Multiply by 4

# Compute the ratio (max_n_features / M)
ratios = [max_n_features_values[i] / M_values[i] for i in range(len(M_values))]
print(M_values)
# Plot
fig, ax = plt.subplots(figsize=(8, 4))
x_positions = np.arange(len(experiments))

# Bars for the ratio
bars = ax.bar(x_positions, ratios, color="skyblue", label="Portion of PCA features")

# Bars for the remaining space (to make total height 1)
ax.bar(x_positions, [1 - r for r in ratios], bottom=ratios, color="orange", label="Remaining random features")

# Add text inside the blue bars (rounded max_n_features)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # Center of the bar
        height / 2,  # Middle of the blue part
        str(round(max_n_features_values[i]/4)),  # Rounded value
        ha="center",
        va="center",
        color="black",
        fontsize=10,
    )

# Formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(experiments, rotation=45, ha="right")
ax.set_ylabel("Proportion")
ax.set_xlabel("M")
ax.legend()
ax.set_ylim(0, 1)  # Ensure bars always reach height 1


plt.tight_layout()
# Show plot
plt.show()
