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
    # sim.visu = True
    sim.visu_type = "qt"
    sim.random_seed = "auto"
    sim.number_of_threads = 4
    sim.progress_bar = True
    sim.output_dir = Path("test01") / f"arf"

    # units
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    deg = gate.g4_units.deg
    Bq = gate.g4_units.Bq

    # options
    radius = 28 * cm
    rad = "lu177"
    colli_type = "melp"
    activity = 1e8 * Bq
    angle_tolerance = 10 * deg

    # visu
    if sim.visu:
        sim.number_of_threads = 1
        activity = 1000 * Bq

    # world etc
    stats = init_sim(sim)

    # set the two spect heads
    spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]
    size = [128 * 2, 128 * 2]
    pth = Path("pth") / "intevo_lu177_v3.pth"
    det_plane1, arf1 = intevo.add_arf_detector(
        sim, "det1", colli_type, size, spacing, pth
    )
    det_plane2, arf2 = intevo.add_arf_detector(
        sim, "det2", colli_type, size, spacing, pth
    )

    # output names
    arf1.output_filename = "projection_1.mhd"
    arf2.output_filename = "projection_2.mhd"

    # compute the gantry rotations
    intevo.rotate_gantry(det_plane1, radius, 0)
    intevo.rotate_gantry(det_plane2, radius, 180)

    # source for test
    add_source_point_test(sim, rad, (det_plane1, det_plane2), activity, angle_tolerance)

    # go
    sim.run()
    print(stats)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
