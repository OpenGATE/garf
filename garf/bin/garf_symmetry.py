#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import click

import garf

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("dataset")
@click.option("--fold", default=6, show_default=True, help="order of the symmetry")
@click.option(
    "--angle-convention",
    default=None,
    help="for old datasets without sidecar: legacy_acos or theta_phi",
)
@click.option("--json", "json_output", default=None, help="also write the result here")
def garf_symmetry(dataset, fold, angle_convention, json_output):
    """
    Azimuthal symmetry of the detector response in a training DATASET: per
    theta bin, amplitude and phase of the FOLD-fold pattern of the detected
    fraction vs phi, and phi0 (orientation of the pattern), as used by the
    "symmetric" features (phi0_deg). Hexagonal holes: fold 6.
    """
    ds = garf.load_training_dataset(dataset, angle_convention=angle_convention)
    r = garf.measure_symmetry(ds.ux, ds.uy, ds.uz, ds.window, fold=fold)
    print(garf.symmetry_table(r))
    if json_output:
        with open(json_output, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    garf_symmetry()
