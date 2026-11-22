#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script visualizes and validates a parametric ARF model against the
original Monte Carlo simulation data.

It takes a training dataset (.root file) and a parametric model (.json file)
as input, and for each energy window, it plots the raw data histogram
overlaid with the fitted "Sum of Gaussians" model curve.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

# Import the data loading function from your library
from garf import load_training_dataset


def sum_of_gaussians(x, p, n_gaussians):
    """
    Calculates the value of a sum of N symmetric Gaussians.

    Args:
        x (np.ndarray): The input values (e.g., theta angles).
        p (list): A flat list of parameters [A1, sigma1, A2, sigma2, ...].
        n_gaussians (int): The number of Gaussians in the sum.

    Returns:
        np.ndarray: The calculated values from the sum of Gaussians model.
    """
    result = np.zeros_like(x, dtype=float)
    for i in range(n_gaussians):
        amplitude = p[2 * i]
        sigma = p[2 * i + 1]
        result += amplitude * np.exp(-(x**2) / (2.0 * sigma**2))
    return result


def plot_parametric_comparison(dataset_path, model_path, n_bins=180, output_path=None):
    """
    Loads data and a parametric model, then plots their comparison.
    """
    # 1. LOAD DATA AND MODEL
    # ===================================================================
    print(f"Loading Monte Carlo dataset from '{dataset_path}'...")
    try:
        # **MODIFIED**: Use the same ROOT loading function as garf_train.py
        data, theta, phi, E, w = load_training_dataset(dataset_path)
        y_train = data[:, 3].astype(int)  # The window 'w' is the target
        x_train = data[:, :3]  # Keep x_train for consistency
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loading parametric model from '{model_path}'...")
    try:
        with open(model_path, "r") as f:
            model_data = json.load(f)
        fitting_results = model_data["fitting_parameters"]
        model_type_str = model_data.get("model_type", "ParametricSumOfNGaussians")
        # Infer n_gaussians from the model type string
        n_gaussians = int("".join(filter(str.isdigit, model_type_str)))
    except Exception as e:
        print(f"Error loading or parsing model JSON: {e}")
        return

    # 2. SETUP PLOTTING
    # ===================================================================
    windows_to_plot = sorted([int(k.split("_")[1]) for k in fitting_results.keys()])
    n_windows = len(windows_to_plot)
    if n_windows == 0:
        print("No fitted windows found in the model file.")
        return

    # Create a grid of subplots
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False
    )
    fig.suptitle(
        f"Parametric Model Fit vs. Monte Carlo Data ({n_gaussians} Gaussians)",
        fontsize=16,
    )

    # 3. PLOT EACH WINDOW
    # ===================================================================
    for i, window_idx in enumerate(windows_to_plot):
        ax = axes[i // n_cols, i % n_cols]

        # --- Get Monte Carlo Data (Ground Truth) ---
        # **CORRECTED**: Use the 'theta' array directly for clarity
        thetas_for_window = np.abs(theta[y_train == window_idx])
        counts, bin_edges = np.histogram(thetas_for_window, bins=n_bins, range=(0, 90))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # --- Get Parametric Model Curve ---
        window_key = f"window_{window_idx}"
        params_dict = fitting_results[window_key]["params"]
        # Reconstruct the flat parameter list [A1, sigma1, ...]
        p_fit = []
        for j in range(n_gaussians):
            gauss_key = f"gaussian_{j+1}"
            p_fit.append(params_dict[gauss_key]["amplitude"])
            p_fit.append(params_dict[gauss_key]["sigma_deg"])

        # Calculate the model's curve at the bin centers
        model_curve = sum_of_gaussians(bin_centers, p_fit, n_gaussians)

        # --- Plot Both ---
        # Plot Monte Carlo data as a step plot (histogram style)
        ax.step(
            bin_centers,
            counts,
            where="mid",
            label="Monte Carlo Data",
            color="royalblue",
            linewidth=1.5,
        )

        # Plot the fitted model as a smooth line
        ax.plot(
            bin_centers,
            model_curve,
            label="Parametric Fit",
            color="darkorange",
            linestyle="--",
            linewidth=2,
        )

        ax.set_title(f"Energy Window {window_idx}")
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    # Clean up empty subplots
    for i in range(n_windows, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols, i % n_cols])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save or show the plot
    if output_path:
        print(f"Saving plot to '{output_path}'...")
        plt.savefig(output_path, dpi=300)

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and compare a parametric ARF model with Monte Carlo data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the training dataset .root file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the parametric model .json file.",
    )
    parser.add_argument(
        "--bins", type=int, default=180, help="Number of bins for the histogram."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output plot image file.",
    )

    args = parser.parse_args()

    plot_parametric_comparison(args.dataset, args.model, args.bins, args.output)
