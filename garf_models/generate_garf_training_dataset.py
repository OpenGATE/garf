#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import click
from garf_models_helpers import *

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--spect", "-s", default="intevo", help="SPECT system: intevo, nm670")
@click.option("--rad", "-r", required=True, help="Radionuclide 177lu, 99tcm etc")
@click.option("--digitizer", "-d", required=True, help="Digitizer version : v1 v2 etc")
@click.option("--n", "-n", default=5e8, help="Number of particles")
@click.option("--rr", default=50, help="Russian Roulette for GARF")
@click.option("--visu", is_flag=True, default=False, help="visu only")
@click.option("--threads", "-t", default=4, help="number of threads")
@click.option(
    "--crystal_size", "-c", default="5/8", help="For nm670 crystal size 3/8 or 5/8"
)
def go(spect, rad, digitizer, n, rr, visu, threads, crystal_size):

    # set the default simulation param
    spect_type = spect
    sim = gate.Simulation()
    arf_basename = f"{spect_type}_{rad}_{digitizer}"
    stats = init_default_garf_simulation(sim, visu, threads)
    sim.output_dir = Path("output") / arf_basename

    # set the SPECT system
    print(f"SPECT type is : {spect_type}")
    head, colli, crystal, colli_type = add_spect_imaging_device(
        sim, spect_type, rad, crystal_size
    )
    print(f"collimator type is : {colli_type}")

    # set the digitizer and the arf
    ew = add_digitizer(sim, spect_type, rad, digitizer, crystal)
    detector_plane, arf = add_arf(sim, spect_type, head, colli_type, ew, rr)

    # visu option
    if visu:
        n = 1e2
        sim.number_of_threads = 1
        sim.output_dir = Path("output") / f"visu"

    # set the source
    Bq = gate.g4_units.Bq
    activity = n * Bq / sim.number_of_threads
    add_source(sim, spect_type, rad, activity, detector_plane)

    # go
    sim.run()

    # print results at the end
    print(stats)

    # print cmd line to train GARF
    pth_json = Path("pth") / "train_arf_v034.json"
    pth = Path("pth") / f"{arf_basename}.pth"
    print(f"garf_train {pth_json} {arf.get_output_path()} {pth}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
