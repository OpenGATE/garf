#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from opengate.tests import utility


def go():

    folder_base = Path("test02")
    folder_ref = folder_base / "reference"
    folder_arf = folder_base / "arf"
    folder_ff = folder_base / "free_flight"

    stats_ref = utility.read_stat_file(folder_ref / "stats.txt")

    is_ok = True
    scaling = 10

    im_name = "projection_1.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_arf / im_name,
            stats_ref,
            tolerance=45,
            ignore_value_data1=0,
            sum_tolerance=11,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )

    im_name = "projection_2.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_arf / im_name,
            stats_ref,
            tolerance=45,
            ignore_value_data1=0,
            sum_tolerance=11,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )

    print()
    im_name = "projection_1.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_ff / im_name,
            stats_ref,
            tolerance=45,
            ignore_value_data1=0,
            sum_tolerance=11,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )

    """im_name = "projection_2.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_ff / im_name,
            stats_ref,
            tolerance=45,
            ignore_value_data1=0,
            sum_tolerance=11,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )"""

    utility.test_ok(is_ok)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
