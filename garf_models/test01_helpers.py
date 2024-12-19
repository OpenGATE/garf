#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.contrib.phantoms.nemaiec as nemaiec
from opengate.sources.base import set_source_rad_energy_spectrum
from digitizers import *
import numpy as np

def init_sim(sim):
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    # world
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = "G4_AIR"
    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.global_production_cuts.all = 1 * mm
    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = f"stats.txt"
    return stats



def add_vox_iec(sim, spacing, data_path):
    mm = gate.g4_units.mm
    # voxelize_iec_phantom -o data/iec_5mm.mha --spacing 5
    print(f"Phantom: IEC voxelized {spacing}mm")
    mhd_filename = data_path / f"iec_{spacing}mm.mha"
    labels_filename = data_path / f"iec_{spacing}mm.json"
    iec = sim.add_volume("Image", "iec")
    iec.image = mhd_filename
    nemaiec.create_material(sim)
    iec.set_materials_from_voxelisation(labels_filename)
    sim.physics_manager.set_production_cut(iec.name, "all", 2 * mm)
    return iec


def add_vox_source(sim, rad, activity, data_path):
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    # voxelize_iec_phantom -o data/iec_1mm.mhd --spacing 1 --output_source data/iec_1mm_activity.mhd -a 1 2 3 4 5 6
    iec_source_filename = data_path / "iec_1mm_activity.mhd"
    source = sim.add_source("VoxelSource", "src")
    source.image = iec_source_filename
    source.position.translation = [0, 35 * mm, 0]
    set_source_rad_energy_spectrum(source, rad)
    source.particle = "gamma"
    _, volumes = nemaiec.get_default_sphere_centers_and_volumes()
    print(f"Volumes are {volumes}")
    source.activity = activity * np.array(volumes).sum()
    print(f"Total activity is {source.activity / Bq}")
    return source

def add_source_point_test(sim, rad, planes, activity, angle_tolerance, head_type='arf'):
    cm = gate.g4_units.cm
    source = sim.add_source("GenericSource", "source_point")
    set_source_rad_energy_spectrum(source, rad)
    source.particle = "gamma"
    source.direction.type = "iso"
    source.position.type = "sphere"
    source.position.radius = 1*cm
    source.direction.acceptance_angle.volumes = [p.name for p in planes]
    source.direction.acceptance_angle.skip_policy = "SkipEvents"
    source.direction.acceptance_angle.intersection_flag = True
    source.direction.acceptance_angle.normal_flag = True
    source.direction.acceptance_angle.normal_vector = [0, 0, -1]
    if head_type == 'spect':
        source.direction.acceptance_angle.normal_vector = [1, 0, 0]
    source.direction.acceptance_angle.normal_tolerance = angle_tolerance
    source.activity = activity
    source.attached_to = 'b'

    b = sim.add_volume('Box', 'b')
    #b.translation = [3*cm, 10*cm, 2*cm]
    b.translation = [3*cm, 0*cm, 2*cm]
    b.material = 'G4_AIR'

    source = sim.add_source("GenericSource", "source_point2")
    set_source_rad_energy_spectrum(source, rad)
    source.particle = "gamma"
    source.direction.type = "iso"
    source.position.type = "sphere"
    source.position.radius = 1*cm
    source.direction.acceptance_angle.volumes = [p.name for p in planes]
    source.direction.acceptance_angle.skip_policy = "SkipEvents"
    source.direction.acceptance_angle.intersection_flag = True
    source.direction.acceptance_angle.normal_flag = True
    source.direction.acceptance_angle.normal_vector = [0, 0, -1]
    if head_type == 'spect':
        source.direction.acceptance_angle.normal_vector = [1, 0, 0]
    source.direction.acceptance_angle.normal_tolerance = angle_tolerance
    source.activity = activity/2