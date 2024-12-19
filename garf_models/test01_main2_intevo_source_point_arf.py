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
    sim.output_dir = Path("test01") / f"arf"

    # units
    mm = gate.g4_units.mm
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

    # set the two spect heads
    spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]
    size = [128 * 2, 128 * 2]
    pth = Path("pth") / "intevo_lu177_v3.pth"
    det_plane1, arf1 = intevo.add_arf_detector(
        sim, radius, 0, size, spacing, colli_type, "detector", 1, pth
    )
    det_plane2, arf2 = intevo.add_arf_detector(
        sim, radius, 180, size, spacing, colli_type, "detector", 2, pth
    )
    # compute the gantry rotations
    intevo.rotate_gantry(det_plane1, radius, 0, initial_rotation="arf")
    intevo.rotate_gantry(det_plane2, radius, 180, initial_rotation="arf")
    det_planes = [det_plane1, det_plane2]

    # source for test
    add_source_point_test(sim, rad, det_planes, activity, angle_tolerance)

    # go
    sim.run()
    print(stats)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
