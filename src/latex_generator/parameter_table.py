import json
import math


def format_number(num, lower_bound, upper_bound, sig_digits=3):
    """
    Format a numeric value with the given number of significant digits.
    If the absolute value is between lower_bound and upper_bound, format as a fixed point;
    otherwise, format in scientific notation (e.g., 'x \\times 10^{y}').
    """
    # Handle the zero case
    if num == 0:
        return "0"

    abs_val = abs(num)
    if lower_bound <= abs_val <= upper_bound:
        # Calculate number of decimals: if the integer part has k digits, we want sig_digits - k decimals.
        k = math.floor(math.log10(abs_val)) + 1
        decimals = max(sig_digits - k, 0)
        return f"${num:.{decimals}f}$"
    else:
        # For scientific notation:
        exp = int(math.floor(math.log10(abs_val)))
        significand = num / (10 ** exp)
        # Format the significand with the given number of significant digits
        significand_str = f"{significand:.{sig_digits}g}"
        return f"${significand_str} \\cdot 10^{{{exp}}}$"


def json_to_latex_table(json_path, variables, lower_bound=1e-3, upper_bound=1e3, sig_digits=3):
    """
    Reads a JSON file and produces a LaTeX table.

    Parameters:
        json_path (str): Path to the JSON file.
        variables (dict): Dictionary where keys are the parameter names to extract from each entry,
                          and values are the corresponding LaTeX label (e.g., {"sigma": "$\\sigma$"}).
        lower_bound (float): Lower bound for using fixed point notation.
        upper_bound (float): Upper bound for using fixed point notation.
        sig_digits (int): Number of significant digits for numerical formatting.

    The JSON file is expected to have a structure similar to:

        {
          "100": {"sigma": 0.00123, "mu": 1.2345, ...},
          "200": {"sigma": 0.00456, "mu": 6.7890, ...},
          ...
        }

    The resulting LaTeX table will have the parameter names (using the provided LaTeX labels)
    in the first column and one column per JSON key (sorted numerically if possible).
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Sort the keys numerically when possible
    def try_convert(key):
        try:
            return float(key)
        except ValueError:
            return key

    sorted_keys = sorted(data.keys(), key=lambda k: try_convert(k))

    # Start constructing the LaTeX table
    latex_lines = []
    n_cols = len(sorted_keys) + 1  # +1 for the variable names column
    col_align = "l" + "c" * len(sorted_keys)  # left-align the first column, center the others
    latex_lines.append("\\begin{tabular}{" + col_align + "}")
    latex_lines.append("\\hline")

    # Header row: leave first cell empty, then list the sorted keys
    header_line = " & " + " & ".join(sorted_keys) + " \\\\"
    latex_lines.append(header_line)
    latex_lines.append("\\hline")

    # For each variable, add a row with the LaTeX label and then the values for each key
    for var, latex_label in variables.items():
        row_values = [latex_label]  # first column: LaTeX formatted variable name
        for key in sorted_keys:
            entry = data.get(key, {})
            value = entry.get(var, "")
            try:
                # Try to convert to float and then format
                num_value = float(value)
                formatted = format_number(num_value, lower_bound, upper_bound, sig_digits)
            except (ValueError, TypeError):
                # If conversion fails, leave as is
                formatted = str(value)
            row_values.append(formatted)
        latex_lines.append(" & ".join(row_values) + " \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")

    # Join the lines and print the LaTeX code
    latex_code = "\n".join(latex_lines)
    print(latex_code)
    return latex_code


# --- Example Usage ---
if __name__ == "__main__":
    # Path to your JSON file
    json_path = "../../res/matrix_500_and_matlab/optimal_parameters_per_M_matrix_conceptor.json"

    # Specify which variables to include and their LaTeX labels
    variables = {
        "noise_std": '$\sigma_\\text{reservoir}$',
        "signal_noise": '$\sigma_\\text{signal}$',
        "aperture_rossler_attractor_2d": '$\\alpha_\\text{Rossler}$',
        "aperture_lorenz_attractor_2d":'$\\alpha_\\text{Lorenz}$',
        "aperture_mackey_glass_2d": '$\\alpha_\\text{Mackey Glass}$',
        "aperture_henon_attractor_2d":'$\\alpha_\\text{Henon}$',
        "beta_W_out": '$\\beta_{W^\\text{out}}$',
        "beta_W": '$\\beta_{W}$',
        "W_in_std": '$\\sigma_{W_{\\text{in}}}$',
        "spectral_radius": '$\\rho(F\\\'G)$',
        "W_sr": "spectral_radius"'$\\rho(W)$',
        'bias': "$b$"
    }

    # Generate the table with default bounds and significant digits
    json_to_latex_table(json_path, variables, lower_bound=0.01, upper_bound=1e3, sig_digits=3)
