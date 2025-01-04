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
    sim.number_of_threads = 4
    sim.progress_bar = True
    sim.output_dir = Path("test02") / f"arf"
    data_path = Path("data")

    # units
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    Bq = gate.g4_units.Bq
    deg = gate.g4_units.deg

    # options
    radius = 28 * cm
    rad = "lu177"
    colli_type = "melp"
    activity = 2e3 * Bq
    angle_tolerance = 10 * deg

    # visu
    if sim.visu:
        sim.number_of_threads = 1
        activity = 0.1 * Bq
        sim.output_dir = Path("test02") / f"visu"

    # world etc
    stats = init_sim(sim)

    # set the two spect heads
    spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]
    size = [128 * 2, 128 * 2]
    pth = Path("pth") / "intevo_lu177_v3_v036.pth"
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
    det_planes = (det_plane1, det_plane2)

    # add voxelized iec
    add_vox_iec(sim, spacing=4, data_path=data_path)

    # add iec voxelized source
    source = add_vox_source(sim, rad, activity, data_path)
    """source.direction.acceptance_angle.volumes = [h.name for h in det_planes]
    source.direction.acceptance_angle.skip_policy = "SkipEvents"
    source.direction.acceptance_angle.intersection_flag = True
    source.direction.acceptance_angle.normal_flag = True
    source.direction.acceptance_angle.normal_vector = [1, 0, 0]
    source.direction.acceptance_angle.normal_tolerance = angle_tolerance
    """

    # go
    sim.run()
    print(stats)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
