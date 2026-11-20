#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import click

import garf

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("pth_filename")
@click.argument("output", required=False)
@click.option(
    "--state-index",
    default=-1,
    show_default=True,
    help="Stored state to convert (garf <= 2.11 always used the last one)",
)
@click.option(
    "--distance-to-crystal",
    "-d",
    type=float,
    default=None,
    help="Distance (mm) from the ARF plane to the middle of the crystal, stored in the model",
)
@click.option("--detector-name", default=None, help="Detector name stored in the model")
def garf_convert(pth_filename, output, state_index, distance_to_crystal, detector_name):
    """
    Convert a legacy garf model (.pth, garf <= 2.11) to the garf v3 format
    (.garf). Only the selected state is kept. The input features are
    legacy_acos (the convention of the training actor of GATE 9 and GATE <= 10.0.2).

    \b
    garf_convert arf_Tc99m_v034.pth                  -> arf_Tc99m_v034.garf
    garf_convert arf_Tc99m_v034.pth out.garf -d 74.625
    """
    pth_filename = Path(pth_filename)
    output = Path(output) if output else pth_filename.with_suffix(garf.FILE_EXTENSION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = garf.ArfModel.load(
            pth_filename, device="cpu", legacy_state_index=state_index
        )
    detector = {}
    if distance_to_crystal is not None:
        detector["distance_to_crystal"] = distance_to_crystal
    if detector_name is not None:
        detector["name"] = detector_name
    if detector:
        model.metadata["detector"] = detector
    model.metadata.setdefault("provenance", {})[
        "converted_by"
    ] = f"garf_convert (garf {garf.model.garf_version()})"
    model.save(output)

    t = model.metadata["training"]
    print(f"Input       : {pth_filename}")
    print(f"Stored epochs {t['stored_epochs']}, selected {t['selected_epoch']}")
    print(f"Features    : {model.features}")
    print(f"Channels    : {model.n_channels} (RR = {model.rr:g})")
    print(f"Architecture: {model.architecture}")
    print(f"Output      : {output} ({output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    garf_convert()
