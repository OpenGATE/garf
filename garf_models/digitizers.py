#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from scipy.spatial.transform import Rotation


def add_intevo_digitizer_lu177_v3(sim, crystal_name, name, spectrum_channel=True):

    # hits
    hits = sim.add_actor("DigitizerHitsCollectionActor", f"hits_{name}")
    hits.attached_to = crystal_name
    hits.output_filename = ""  # No output
    hits.attributes = [
        "PostPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "PostStepUniqueVolumeID",
        "GlobalTime",
    ]

    # singles
    singles = sim.add_actor("DigitizerAdderActor", f"singles_{name}")
    singles.attached_to = crystal_name
    singles.input_digi_collection = hits.name
    # sc.policy = "EnergyWeightedCentroidPosition"
    singles.policy = "EnergyWinnerPosition"
    singles.output_filename = ""  # No output
    singles.group_volume = None

    # efficiency actor
    eff = sim.add_actor("DigitizerEfficiencyActor", f"singles_{name}_eff")
    eff.attached_to = crystal_name
    eff.input_digi_collection = singles.name
    eff.efficiency = 0.86481  # FIXME probably wrong, to evaluate
    eff.efficiency = 1.0
    eff.output_filename = ""  # No output

    # energy blur
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV
    ene_blur = sim.add_actor("DigitizerBlurringActor", f"singles_{name}_eblur")
    ene_blur.output_filename = ""
    ene_blur.attached_to = crystal_name
    ene_blur.input_digi_collection = eff.name
    ene_blur.blur_attribute = "TotalEnergyDeposit"
    ene_blur.blur_method = "Linear"
    ene_blur.blur_resolution = 0.13
    ene_blur.blur_reference_value = 80 * keV
    ene_blur.blur_slope = -0.09 * 1 / MeV

    # spatial blurring
    mm = gate.g4_units.mm
    spatial_blur = sim.add_actor(
        "DigitizerSpatialBlurringActor", f"singles_{name}_sblur"
    )
    spatial_blur.output_filename = ""
    spatial_blur.attached_to = crystal_name
    spatial_blur.input_digi_collection = ene_blur.name
    spatial_blur.blur_attribute = "PostPosition"
    spatial_blur.blur_fwhm = 3.9 * mm
    spatial_blur.keep_in_solid_limits = True

    # energy windows
    singles_ene_windows = sim.add_actor(
        "DigitizerEnergyWindowsActor", f"{name}_energy_window"
    )
    channels = [
        {"name": f"spectrum", "min": 3 * keV, "max": 515 * keV},
        {"name": f"scatter1_{name}", "min": 96 * keV, "max": 104 * keV},
        {"name": f"peak113_{name}", "min": 104.52 * keV, "max": 121.48 * keV},
        {"name": f"scatter2_{name}", "min": 122.48 * keV, "max": 133.12 * keV},
        {"name": f"scatter3_{name}", "min": 176.46 * keV, "max": 191.36 * keV},
        {"name": f"peak208_{name}", "min": 192.4 * keV, "max": 223.6 * keV},
        {"name": f"scatter4_{name}", "min": 224.64 * keV, "max": 243.3 * keV},
    ]
    if not spectrum_channel:
        channels.pop(0)
    singles_ene_windows.attached_to = crystal_name
    singles_ene_windows.input_digi_collection = spatial_blur.name
    singles_ene_windows.channels = channels
    singles_ene_windows.output_filename = ""  # No output

    # projection
    proj = sim.add_actor("DigitizerProjectionActor", f"{name}_projection")
    proj.attached_to = crystal_name
    proj.input_digi_collections = [x["name"] for x in channels]
    proj.spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]
    proj.size = [128 * 2, 128 * 2]
    proj.output_filename = "projection.mhd"
    proj.origin_as_image_center = True

    # plane orientation
    proj.detector_orientation_matrix = Rotation.from_euler(
        "yx", (90, 90), degrees=True
    ).as_matrix()

    return proj
