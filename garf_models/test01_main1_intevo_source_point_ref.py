#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.contrib.spect.siemens_intevo as intevo
from pathlib import Path
from digitizers import *
from test01_helpers import init_sim, add_source_point_test


def go():

    # create the simulation
    sim = gate.Simulation()

    # main options
    #sim.visu = True
    sim.visu_type = "qt"
    sim.random_seed = "auto"
    sim.number_of_threads = 4
    sim.progress_bar = True
    sim.output_dir = Path("test01") / f"reference"

    # units
    cm = gate.g4_units.cm
    deg = gate.g4_units.deg

    # options
    radius = 28 * cm
    rad = "lu177"
    colli_type = "melp"
    activity = 4e5
    angle_tolerance = 10 * deg

    # visu
    if sim.visu:
        sim.number_of_threads = 1
        activity = 100

    # world etc
    stats = init_sim(sim)

    # set the spect head
    head1, colli1, crystal1 = intevo.add_spect_head(
        sim, "spect1", collimator_type=colli_type, debug=sim.visu == True
    )
    proj1 = add_intevo_digitizer_lu177_v3(
        sim, crystal1, f"digitizer1", spectrum_channel=False
    )
    proj1.output_filename = "projection_1.mhd"

    head2, colli2, crystal2 = intevo.add_spect_head(
        sim, "spect2", collimator_type=colli_type, debug=sim.visu == True
    )
    proj2 = add_intevo_digitizer_lu177_v3(
        sim, crystal2, f"digitizer2", spectrum_channel=False
    )
    proj2.output_filename = "projection_2.mhd"

    # rotate
    intevo.rotate_gantry(head1, radius, 0, initial_rotation='spect')
    intevo.rotate_gantry(head2, radius, 180, initial_rotation='spect')

    # source for test
    add_source_point_test(sim, rad, (head1, head2), activity, angle_tolerance, head_type='spect')

    # go
    sim.run()
    print(stats)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
