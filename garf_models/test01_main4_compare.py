#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate.contrib.spect.siemens_intevo as intevo
from pathlib import Path
from digitizers import *
from test01_helpers import init_sim, add_source_point_test
from opengate.tests import utility


def go():

    folder_base = Path('test01')
    folder_ref = folder_base / "reference"
    folder_arf = folder_base / "arf"
    folder_ff = folder_base / "free_flight"

    stats_ref = utility.read_stat_file(folder_ref / "stats.txt")

    is_ok = True

    im_name = "projection_1.mhd"
    is_ok = utility.assert_images(
                folder_ref / im_name,
                folder_arf / im_name,
                stats_ref,
                tolerance=38,
                ignore_value_data1=0,
                sum_tolerance=8,
                axis="x",
            ) and is_ok

    im_name = "projection_2.mhd"
    is_ok = utility.assert_images(
                folder_ref / im_name,
                folder_arf / im_name,
                stats_ref,
                tolerance=38,
                ignore_value_data1=0,
                sum_tolerance=8,
                axis="x",
            ) and is_ok
    utility.test_ok(is_ok)



# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
