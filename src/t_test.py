from scipy.stats import ttest_ind
import os
import json
import numpy as np


def perform_hypothesis_tests(folder_path, rfc_type1, rfc_type2, alpha=0.05, repetitions=30):
    """
    Perform one-sided hypothesis tests for NRMSE values between two RFC types.

    Args:
        folder_path (str): Path to the folder containing the NRMSE results JSON files.
        rfc_type1 (str): First RFC type to compare.
        rfc_type2 (str): Second RFC type to compare.
        alpha (float): Significance level for the hypothesis test (default: 0.05).
        repetitions (int): The specific number of repetitions to filter files for comparison.
    """
    # Dictionaries to store data for RFC types
    rfc_data = {rfc_type1: {}, rfc_type2: {}}

    # Load NRMSE data for the specified RFC types
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

            # Store data for the requested RFC types
            if rfc_type in rfc_data:
                for M, nrmse_values in data.items():
                    M = float(M)
                    if M not in rfc_data[rfc_type]:
                        rfc_data[rfc_type][M] = []
                    rfc_data[rfc_type][M].extend(nrmse_values)
    # Perform hypothesis tests for each shared M
    shared_M = set(rfc_data[rfc_type1].keys()).intersection(rfc_data[rfc_type2].keys())
    if not shared_M:
        print("No shared M values found between the two RFC types.")
        return

    print(f"Hypothesis Testing (RFC Type 1: {rfc_type1}, RFC Type 2: {rfc_type2})")
    print(f"Significance Level (alpha): {alpha}\n")
    for M in sorted(shared_M):
        values1 = rfc_data[rfc_type1][M]
        values2 = rfc_data[rfc_type2][M]

        # Perform a one-sided t-test not equal var
        t_stat, p_value = ttest_ind(values1, values2, alternative='less', equal_var=False)

        # Check if we reject the null hypothesis
        reject_null = p_value < alpha
        conclusion = "Reject H0" if reject_null else "Fail to reject H0"

        print(f"M = {M}")
        print(f"  RFC Type 1 Mean NRMSE: {np.mean(values1):.4f}, variance: {np.var(values1)}")
        print(f"  RFC Type 2 Mean NRMSE: {np.mean(values2):.4f}, variance: {np.var(values2)}")
        print(f"  T-Statistic: {t_stat:.4f}, P-Value: {p_value:.6f}")
        print(f"  Conclusion: {conclusion}\n")

# Example usage
folder_path = '../res/optimize_different_M_2'  # Replace with your folder path
perform_hypothesis_tests(folder_path, 'PCARFC', 'base', alpha=0.01, repetitions=30)
