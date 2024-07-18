#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
def pytorch_info():
    """
    \b
    Print pytorch version
    """

    print("pytorch version", torch.__version__)

    cuda_mode = torch.cuda.is_available()
    if cuda_mode:
        print("CUDA GPU mode is available")
        print("CUDA version:", torch.version.cuda)
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device counts:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA GPU")

    mps_mode = torch.backends.mps.is_available()
    if mps_mode:
        print("MPS GPU mode is available")
    else:
        print("No MPS GPU")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pytorch_info()
