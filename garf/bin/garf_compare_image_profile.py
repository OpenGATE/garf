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
    "--events",
    "-e",
    default=float(1),
    help="Scale the image2 by this value before comparing",
)
@click.option("--islice", "-s", default=int(64), help="Image slice for the profile")
@click.option("--wslice", "-w", default=int(3), help="Slice width (to smooth)")
def garf_compare_image_profile(image1_mhd, image2_mhd, islice, events, wslice):
    # Load image
    img_ref = sitk.ReadImage(image1_mhd)
    img = sitk.ReadImage(image2_mhd)
    events = float(events)
    islice = int(islice)
    wslice = int(wslice)

    # Scale data to the ref nb of particles
    img = img * events

    # Get the pixels values as np array
    data_ref = sitk.GetArrayFromImage(img_ref).astype(float)
    data = sitk.GetArrayFromImage(img).astype(float)

    # Sometimes not same nb of slices -> crop the data_ref
    if len(data_ref) > len(data):
        data_ref = data_ref[0 : len(data), :, :]

    # Criterion1: global counts in every windows
    s_ref = np.sum(data_ref, axis=(1, 2))
    s = np.sum(data, axis=(1, 2))

    print("Ref:     Singles/Scatter/Peak1/Peak2: {}".format(s_ref))
    print("Img:     WARNING/Scatter/Peak1/Peak2: {}".format(s))
    print(
        "% diff : WARNING/Scatter/Peak1/Peak2: {}".format((s - s_ref) / s_ref * 100.0)
    )

    # Profiles
    # data image: !eee!Z,Y,X
    p_ref = np.mean(data_ref[:, islice - wslice : islice + wslice - 1, :], axis=1)
    p = np.mean(data[:, islice - wslice : islice + wslice - 1, :], axis=1)
    x = np.arange(0, 128, 1)

    nb_ene = len(data)
    print("Nb of energy windows: ", nb_ene)

    if nb_ene == 3:  # Tc99m
        win = ["WARNING", "Scatter", "Peak 140"]

    if nb_ene == 6:  # In111
        win = ["WARNING", "Scatter1", "Peak171", "Scatter2", "Scatter3", "Peak245"]

    if nb_ene == 7:  # Lu177
        win = [
            "WARNING",
            "Scatter1",
            "Peak113",
            "Scatter2",
            "Scatter3",
            "Peak208",
            "Scatter4",
        ]

    if nb_ene == 8:
        win = [
            "WARNING",
            "Scatter1",
            "Peak364",
            "Scatter2",
            "Scatter3",
            "Scatter4",
            "Peak637",
            "Peak722",
        ]

    fig, ax = plt.subplots(ncols=nb_ene - 1, nrows=1, figsize=(35, 5))

    i = 1
    vmax = np.max(p_ref[1:, :])
    vmax = np.max(p[1:, :])
    print("Max value in ref image for the scale : {}".format(vmax))

    fs = 12

    plt.rc("font", size=fs)
    while i < nb_ene:
        a = ax[i - 1]

        a.plot(x, p_ref[i], "g", label="Analog", alpha=0.5, linewidth=2.0)
        a.plot(x, p[i], "k--", label="ARF", alpha=0.9, linewidth=1.0)
        a.set_title(win[i], fontsize=fs + 5)
        a.legend(loc="best")
        # a.labelsize = 40
        a.tick_params(labelsize=fs)
        # a.set_ylim([0, vmax])
        i += 1

    plt.suptitle("Compare " + image1_mhd + " vs " + image2_mhd + " w=" + str(wslice))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("output.pdf")
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    garf_compare_image_profile()
