#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import garf
import click
import numpy as np
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename_pth")
def garf_nn_info(filename_pth):
    """
    \b
    Display information about the NN stored in the pth file.
    <FILENAME_PTH> : nn file model in pth (pytorch) format
    """
    nn, model = garf.load_nn(filename_pth)
    p = nn["model_data"]
    garf.print_nn_params(p)

    loss_values = p["loss_values"]
    x = np.arange(0, len(loss_values))
    plt.plot(x, loss_values)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_nn_info()
