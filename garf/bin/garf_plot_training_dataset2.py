#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import garf
import matplotlib.pyplot as plt
import numpy as np
import click

# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("data_file")
@click.option(
    "--sample-size", default=5000, help="Number of non-detected (w=0) events to plot."
)
def garf_plot_training_dataset_2d(data_file, sample_size):
    """
    \b
    Display 2D scatter plots of the training dataset to show relationships
    between angles, energy, and the final energy window.

    <DATA_FILE> : dataset in root format
    """
    print(f"Loading data from '{data_file}'")
    data, theta, phi, E, w = garf.load_training_dataset(data_file)
    print("Data loaded. Preparing plots...")

    # Find the unique window IDs and assign colors
    window_ids = np.unique(w)
    colors = plt.cm.viridis(np.linspace(0, 1, len(window_ids)))

    # Create a 1x3 figure for the plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Sub-sample the dominant class for clarity ---
    # Separate data by window ID
    data_by_window = {win_id: data[w == win_id] for win_id in window_ids}

    # Sub-sample the window=0 class if it's too large
    if 0 in data_by_window and len(data_by_window[0]) > sample_size:
        print(
            f"Sub-sampling non-detected (window=0) class from {len(data_by_window[0])} to {sample_size} points for clarity."
        )
        indices = np.random.choice(
            data_by_window[0].shape[0], sample_size, replace=False
        )
        data_by_window[0] = data_by_window[0][indices]

    # --- Create the plots ---
    plot_titles = ["Theta vs. Phi", "Energy vs. Theta", "Energy vs. Phi"]
    plot_vars = [(1, 0), (0, 2), (1, 2)]  # (x_col_idx, y_col_idx) from `data` array

    for i, p_ax in enumerate(ax):
        for win_id, color in zip(window_ids, colors):
            if win_id not in data_by_window:
                continue

            subset = data_by_window[win_id]
            x_data = subset[:, plot_vars[i][0]]
            y_data = subset[:, plot_vars[i][1]]

            # For Energy plots, convert to keV
            if plot_vars[i][1] == 2:  # Energy is the Y-axis
                y_data = y_data * 1000

            p_ax.scatter(
                x_data,
                y_data,
                color=color,
                label=f"Window {int(win_id)}",
                alpha=0.5,
                s=5,
            )

        p_ax.set_title(plot_titles[i])
        p_ax.set_xlabel(f"{['Theta', 'Phi', 'Energy (keV)'][plot_vars[i][0]]}")
        p_ax.set_ylabel(f"{['Theta', 'Phi', 'Energy (keV)'][plot_vars[i][1]]}")
        p_ax.legend()
        p_ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_plot_training_dataset_2d()
