#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.contrib.spect.ge_discovery_nm670 as nm670
import opengate.contrib.spect.siemens_intevo as intevo
from opengate.sources.base import get_rad_gamma_spectrum
from opengate.exception import fatal
from digitizers import *

all_spects = ("intevo", "nm670")

all_collimators = {
    "intevo": {"tc99m": "lehr", "lu177": "melp", "in111": "melp", "i131": "he"},
    "nm670": {"tc99m": "lehr", "lu177": "megp", "in111": "megp", "i131": "hegp"},
}

all_digitizers = {
    "intevo": {
        "v1": {
            "tc99m": intevo.add_digitizer_tc99m,
            "lu177": intevo.add_digitizer_lu177,
        },
        "v2": {"tc99m": intevo.add_digitizer_tc99m_v2},
        "v3": {"lu177": add_intevo_digitizer_lu177_v3},
    },
    "nm670": {
        "v1": {
            "tc99m": nm670.add_digitizer_tc99m,
            "lu177": nm670.add_digitizer_lu177,
        },
        "v2": {"tc99m": nm670.add_digitizer_tc99m_v2},
    },
}


def init_default_garf_simulation(sim, visu, threads):
    # main options
    sim.visu = visu
    sim.visu_type = "qt"
    sim.number_of_threads = threads
    sim.progress_bar = True
    sim.store_json_archive = True
    sim.store_input_files = False
    sim.json_archive_filename = "simu.json"

    # units
    mm = gate.g4_units.mm
    m = gate.g4_units.m

    # world
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = "G4_AIR"

    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.global_production_cuts.all = 1 * mm

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = "stats.txt"
    return stats


def get_collimator_from_rad(spect, rad):
    spect = spect.lower()
    rad = rad.lower()
    if spect not in all_collimators:
        fatal(f'Unknown spect system "{spect}", known are: {all_collimators.keys()} ')
    if rad not in all_collimators[spect]:
        fatal(
            f'Unknown radionuclide "{rad}", known are: {all_collimators[spect].keys()} '
        )
    return all_collimators[spect][rad]


def add_spect_imaging_device(sim, spect, rad, crystal_size):
    if spect not in all_spects:
        fatal(f'Unknown spect system "{spect}", known are: {all_spects} ')
    colli_type = get_collimator_from_rad(spect, rad)
    mm = gate.g4_units.mm
    head, colli, crystal = None, None, None
    if spect == "intevo":
        head, colli, crystal = intevo.add_spect_head(
            sim, "spect", collimator_type=colli_type, debug=sim.visu == True
        )
        intevo.set_head_orientation(head, colli_type, radius=0 * mm)
    if spect == "nm670":
        head, colli, crystal = nm670.add_spect_head(
            sim,
            "spect",
            collimator_type=colli_type,
            debug=sim.visu == True,
            crystal_size=crystal_size,
        )

    print("Head position", head.translation)
    return head, colli, crystal, colli_type


def add_digitizer(sim, spect, rad, digitizer, crystal):
    if spect not in all_digitizers:
        fatal(
            f'Unknown digitizer spect system "{spect}", known are: {all_digitizers.keys()} '
        )
    digitizers = all_digitizers[spect]
    if digitizer not in digitizers:
        fatal(f'Unknown digitizer "{digitizer}", known are: {digitizers.keys()} ')
    digitizers = digitizers[digitizer]
    if rad not in digitizers:
        fatal(
            f'Unknown digitizer "{digitizer}" for radionuclide "{rad}", known are: {digitizers.keys()} '
        )

    # create the digitizer
    f = digitizers[rad]

    # digitizer
    f(sim, crystal, f"digitizer")

    # special case for spectrum channel (to remove)
    proj = sim.get_actor(f"digitizer_projection")
    proj.write_to_disk = False
    ew = sim.get_actor(f"digitizer_energy_window")
    channels = ew.channels
    c = []
    for channel in channels:
        if channel.name != "spectrum":
            c.append(channel)
    ew.channels = c
    proj.input_digi_collections = [c["name"] for c in ew.channels]

    return ew


def add_arf(sim, spect, head, colli_type, ew, rr):
    detector_plane = None
    arf = None
    if spect not in all_spects:
        fatal(f'Unknown spect system "{spect}", known are: {all_spects} ')
    if spect == "nm670":
        detector_plane, arf = nm670.add_actor_for_arf_training_dataset(
            sim, head, colli_type, ew, rr=rr
        )
    if spect == "intevo":
        detector_plane, arf = intevo.add_actor_for_arf_training_dataset(
            sim, colli_type, ew, rr=rr
        )
    print("Plane position", detector_plane.translation)
    return detector_plane, arf


def add_source(sim, spect, rad, activity, detector_plane):
    if spect not in all_spects:
        fatal(f'Unknown spect system "{spect}", known are: {all_spects} ')
    MeV = gate.g4_units.MeV
    max_energy = get_rad_gamma_spectrum(rad).energies[-1] * 1.05
    print(f"Max energy for {rad} is {max_energy} MeV")
    source = None
    if spect == "nm670":
        source = nm670.add_source_for_arf_training_dataset(
            sim, "src", activity, detector_plane, 0.01 * MeV, max_energy
        )
    if spect == "intevo":
        source = intevo.add_source_for_arf_training_dataset(
            sim, "src", activity, detector_plane, 0.01 * MeV, max_energy
        )
    print("Source position", source.position.translation)
    return source
