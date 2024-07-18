#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import garf
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import uproot
import ntpath
import click

# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("data")
@click.option("-n", default=1e4, help="Number of values to plot.")
def garf_plot_training_dataset(data, n):
    """
    \b
    Display info about the training dataset.
    <DATA> : dataset in root format
    """
    filename = data
    data, x, y, theta, phi, E = garf.load_test_dataset(filename)

    print("Number of values: ", len(data))

    n = int(n)
    x = x[:n]
    y = y[:n]
    theta = theta[:n]
    phi = phi[:n]
    E = E[:n]

    print("Plot with ", len(x), "values.")

    ## histo binning
    b = 100

    f, ax = plt.subplots(2, 2, figsize=(10, 10))

    n, bins, patches = ax[0, 0].hist(theta, b, density=True, facecolor="g", alpha=0.35)
    n, bins, patches = ax[0, 1].hist(phi, b, density=True, facecolor="g", alpha=0.35)
    n, bins, patches = ax[1, 0].hist(
        E * 1000, b, density=True, facecolor="b", alpha=0.35
    )
    ax[1, 1].scatter(x, y, color="r", alpha=0.35, s=1)

    ax[0, 0].set_xlabel("Theta angle (deg)")
    ax[0, 1].set_xlabel("Phi angle (deg)")
    ax[1, 0].set_xlabel("Energy (keV)")
    ax[1, 1].set_xlabel("X")
    ax[1, 1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_plot_training_dataset()
