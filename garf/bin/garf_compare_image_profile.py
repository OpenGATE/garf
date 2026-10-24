#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import click

# -----------------------------------------------------------------------------
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("image1_mhd")
@click.argument("image2_mhd")
@click.option(
    "--scaling",
    "-s",
    default=1.0,
    help="Scale the image2 by this value before comparing",
)
@click.option(
    "--islice", "-i", default=None, help="Image slice for the profile (middle if None)"
)
@click.option("--wslice", "-w", default=int(3), help="Slice width (to smooth)")
@click.option("--output", "-o", default="output.pdf", help="output")
@click.option(
    "-rmf",
    default=False,
    is_flag=True,
    help="Remove first slice of ref data (hit slice)",
)
def garf_compare_image_profile(
    image1_mhd, image2_mhd, islice, scaling, wslice, rmf, output
):
    # Load image
    img_ref = sitk.ReadImage(image1_mhd)
    img = sitk.ReadImage(image2_mhd)
    scaling = float(scaling)
    wslice = int(wslice)

    # slice
    if islice is None:
        islice = int(img.GetSize()[0] / 2)
    else:
        islice = int(islice)

    # Get the pixels values as np array
    data_ref = sitk.GetArrayFromImage(img_ref).astype(float)
    data = sitk.GetArrayFromImage(img).astype(float)

    # Scale data to the ref nb of particles
    data = data * scaling

    print(f"Reference image shape : {data_ref.shape}")
    print(f"Test image shape      : {data.shape}")

    # Sometimes not same nb of slices -> crop the data_ref
    if len(data_ref) > len(data):
        data_ref = data_ref[0 : len(data), :, :]

    # Remove first slice ?
    if rmf:
        data_ref = data_ref[1:, :, :]

    # Criterion1: global counts in every windows
    s_ref = np.sum(data_ref, axis=(1, 2))
    s = np.sum(data, axis=(1, 2))
    ratio = (s - s_ref) / s_ref * 100.0

    # global counts
    print(f"Global counts, reference : {s_ref}")
    print(f"Global counts, test image: {s}")
    print(f"Global counts, % diff    : {ratio} %")

    # Profiles
    p_ref = np.mean(data_ref[:, islice - wslice : islice + wslice - 1, :], axis=1)
    p = np.mean(data[:, islice - wslice : islice + wslice - 1, :], axis=1)
    x = np.arange(0, data.shape[1], 1)

    # max
    vmax_ref = np.max(p_ref[1:, :])
    vmax = np.max(p[1:, :])
    print(f"Max value in ref image  : {vmax_ref}")
    print(f"Max value in test image : {vmax}")

    # nb of energy windows
    nb_ene = len(data)
    print("Nb of energy windows: ", nb_ene)
    win = [f"win {i}" for i in np.arange(nb_ene)]

    # figure
    fig, ax = plt.subplots(ncols=nb_ene, nrows=1, figsize=(35, 5))
    fs = 12
    plt.rc("font", size=fs)
    for i in range(nb_ene):
        a = ax[i]
        a.plot(x, p_ref[i], "g", label="Analog", alpha=0.5, linewidth=2.0)
        a.plot(x, p[i], "k--", label="ARF", alpha=0.9, linewidth=1.0)
        a.set_title(win[i], fontsize=fs + 5)
        a.legend(loc="best")
        a.tick_params(labelsize=fs)
        i += 1

    plt.suptitle("Compare " + image1_mhd + " vs " + image2_mhd + " w=" + str(wslice))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output)
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_compare_image_profile()
