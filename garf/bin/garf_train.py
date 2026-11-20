#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import click

import garf

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("param")
@click.argument("data")
@click.argument("output")
@click.option(
    "--angle-convention",
    type=click.Choice(["legacy_acos", "theta_phi"]),
    default=None,
    help="Only for old datasets (Theta, Phi): legacy_acos for GATE 9 and "
    "GATE <= 10.0.2, theta_phi for GATE 10.0.3 to 10.1.2",
)
@click.option("--quiet", "-q", is_flag=True, help="No output during the training")
def garf_train(param, data, output, angle_convention, quiet):
    """
    \b
    Train an ARF neural network from a training dataset.
    PARAM  : JSON file with the training parameters, e.g.
             {"H": 200, "L": 3, "epochs": 100, "rr": 50}
             (see garf.TrainingParameters for all of them)
    DATA   : training dataset (ROOT, from the GATE ARFTrainingDatasetActor)
    OUTPUT : model file (.garf)
    """
    params = garf.TrainingParameters.from_dict(json.loads(Path(param).read_text()))
    dataset = garf.load_training_dataset(data, angle_convention=angle_convention)
    if not quiet:
        print(f"Training dataset {data}: {dataset.summary()}")
    model = garf.train(dataset, params, verbose=not quiet)
    model.save(output)
    if not quiet:
        print(f"Model saved to {output}")


if __name__ == "__main__":
    garf_train()
