#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from opengate.tests import utility


def go():

    folder_base = Path("test01")
    folder_ref = folder_base / "reference"
    folder_arf = folder_base / "arf"
    folder_ff = folder_base / "free_flight"
    stats_ref = utility.read_stat_file(folder_ref / "stats.txt")
    is_ok = True

    print(f"Ref vs ARF (projection1) =======================")
    im_name = "projection_1.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_arf / im_name,
            stats_ref,
            sum_tolerance=9,
            axis="x",
            test_sad=False,
            sad_profile_tolerance=10,
        )
        and is_ok
    )

    print()
    print(f"Ref vs ARF (projection2) =======================")
    im_name = "projection_2.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_arf / im_name,
            stats_ref,
            test_sad=False,
            sum_tolerance=9,
            axis="x",
            sad_profile_tolerance=10,
        )
        and is_ok
    )

    print()
    print(f"Ref vs FF (projection1) =======================")
    im_name = "projection_1.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_ff / im_name,
            stats_ref,
            test_sad=False,
            sum_tolerance=9,
            axis="x",
            sad_profile_tolerance=10,
        )
        and is_ok
    )
    print()
    print(f"Ref vs FF (projection2) =======================")
    im_name = "projection_2.mhd"
    is_ok = (
        utility.assert_images(
            folder_ref / im_name,
            folder_ff / im_name,
            stats_ref,
            test_sad=False,
            sum_tolerance=9,
            axis="x",
            sad_profile_tolerance=10,
        )
        and is_ok
    )

    print()
    print(f"ARF vs FF (projection2) =======================")
    im_name = "projection_1.mhd"
    is_ok = (
        utility.assert_images(
            folder_arf / im_name,
            folder_ff / im_name,
            stats_ref,
            test_sad=False,
            sum_tolerance=1,
            axis="x",
            sad_profile_tolerance=3,
        )
        and is_ok
    )

    utility.test_ok(is_ok)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
