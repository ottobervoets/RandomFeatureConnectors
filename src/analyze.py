import pandas as pd
import numpy as np
import plotly.graph_objects as go


def process_and_plot_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate mean_sinus and mean_discrete
    # df['mean_sinus'] = df[['pattern 0 mean', 'pattern 1 mean']].mean(axis=1)
    # df['mean_discrete'] = df[['pattern 2 mean', 'pattern 3 mean']].mean(axis=1)

    # Calculate pooled standard deviation for mean_sinus and mean_discrete
    # df['std_sinus'] = np.sqrt(
    #     ((df['pattern 0 std'] ** 2 + df['pattern 1 std'] ** 2) / 2) / df['n_rep']
    # )
    # df['std_discrete'] = np.sqrt(
    #     ((df['pattern 2 std'] ** 2 + df['pattern 3 std'] ** 2) / 2) / df['n_rep']
    # )

    # Sort the dataframes by mean_sinus and mean_discrete
    df = df[df['M'] == 1000]
    df_sorted = df.sort_values(by='nrmse')


    # Plot the sorted dataframes in a scrollable format
    plot_scrollable_table(df_sorted, title="DataFrame sorted by mean_nrmse")


def plot_scrollable_table(df, title="Scrollable Table"):
    # Calculate appropriate column widths based on content length
    max_width = 400
    col_widths = [min(max_width, max(df[col].astype(str).map(len).max(), len(col) * 10)) for col in df.columns]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left',
                    height=40,  # Increase header height for readability
                    font=dict(size=14),
                    line=dict(width=1)),
        cells=dict(values=[df[col].round(5).astype(str).apply(lambda x: x.replace(" ", "\n")) for col in df.columns],
                   # Wrap content
                   fill_color='lavender',
                   align='left',
                   height=30,
                   line=dict(width=1))
    )])

    # Set the column width to accommodate wider headers and content
    fig.update_layout(
        title=title,
        height=600,  # Scrollable height
        width=sum(col_widths) + 50,  # Total width accommodating column widths
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.show()


process_and_plot_csv("../res/optimize_different_M_2/" + "2025-01-06 16_26_39.csv")
