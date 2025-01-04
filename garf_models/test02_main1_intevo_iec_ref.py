#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.contrib.spect.siemens_intevo as intevo
from pathlib import Path
from digitizers import *
from test01_helpers import add_vox_iec, add_vox_source, init_sim


def go():

    # create the simulation
    sim = gate.Simulation()

    # main options
    # sim.visu = True
    sim.visu_type = "qt"
    sim.random_seed = "auto"
    sim.number_of_threads = 8
    sim.progress_bar = True
    sim.output_dir = Path("test02") / f"reference"
    data_path = Path("data")

    # units
    cm = gate.g4_units.cm
    Bq = gate.g4_units.Bq

    # options
    radius = 28 * cm
    rad = "lu177"
    colli_type = "melp"
    activity = 2e4 * Bq

    # visu
    if sim.visu:
        sim.number_of_threads = 1
        activity = 0.1 * Bq
        sim.output_dir = Path("test02") / f"visu"

    # world etc
    stats = init_sim(sim)

    # set the spect head
    head1, colli1, crystal1 = intevo.add_spect_head(
        sim, "spect1", collimator_type=colli_type, debug=sim.visu == True
    )
    proj1 = add_intevo_digitizer_lu177_v3(
        sim, crystal1, f"digitizer1", spectrum_channel=False
    )

    head2, colli2, crystal2 = intevo.add_spect_head(
        sim, "spect2", collimator_type=colli_type, debug=sim.visu == True
    )
    proj2 = add_intevo_digitizer_lu177_v3(
        sim, crystal2, f"digitizer2", spectrum_channel=False
    )

    # output names
    proj1.output_filename = "projection_1.mhd"
    proj2.output_filename = "projection_2.mhd"

    # rotate
    intevo.rotate_gantry(head1, radius, 0)
    intevo.rotate_gantry(head2, radius, 180)

    # add voxelized iec
    add_vox_iec(sim, spacing=4, data_path=data_path)

    # add iec voxelized source
    add_vox_source(sim, rad, activity, data_path)

    # go
    sim.run()
    print(stats)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
