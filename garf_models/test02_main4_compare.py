#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from opengate.tests import utility
from garf.helpers import plot_spect_projection


def go():

    folder_base = Path("test02")
    folder_ref = folder_base / "reference"
    folder_arf = folder_base / "arf"
    folder_ff = folder_base / "free_flight"
    stats_ref = utility.read_stat_file(folder_ref / "stats.txt")

    is_ok = True
    scaling = 10

    # ref vs ARF proj1
    print(f"Ref vs ARF (projection1) =======================")
    im_name = "projection_1.mhd"
    f1 = folder_ref / im_name
    f2 = folder_arf / im_name
    is_ok = (
        utility.assert_images(
            f1,
            f2,
            stats_ref,
            test_sad=False,
            sum_tolerance=15,
            scaleImageValuesFactor=scaling,
            axis="x",
            sad_profile_tolerance=20,
        )
        and is_ok
    )

    o = folder_arf / "compare_1.pdf"
    plt = plot_spect_projection(f1, f2, scaling=10, islice=117, wslice=3)
    plt.savefig(o)
    print(f"Profile figure : ", o)

    # ref vs ARF proj2
    print()
    print(f"Ref vs ARF (projection2) =======================")
    im_name = "projection_2.mhd"
    f1 = folder_ref / im_name
    f2 = folder_arf / im_name
    is_ok = (
        utility.assert_images(
            f1,
            f2,
            stats_ref,
            test_sad=False,
            sum_tolerance=15,
            scaleImageValuesFactor=scaling,
            axis="x",
            sad_profile_tolerance=20,
        )
        and is_ok
    )

    o = folder_arf / "compare_2.pdf"
    plt = plot_spect_projection(f1, f2, scaling=10, islice=117, wslice=3)
    plt.savefig(o)
    print(f"Profile figure : ", o)

    # ref vs FF proj1
    print()
    print(f"Ref vs FF (projection1) =======================")
    im_name = "projection_1.mhd"
    f1 = folder_ref / im_name
    f2 = folder_ff / im_name
    is_ok = (
        utility.assert_images(
            f1,
            f2,
            stats_ref,
            test_sad=False,
            sum_tolerance=70,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )

    o = folder_ff / "compare_1.pdf"
    plt = plot_spect_projection(f1, f2, scaling=10, islice=117, wslice=3)
    plt.savefig(o)
    print(f"Profile figure : ", o)

    print()
    print(f"Ref vs FF (projection1) =======================")
    im_name = "projection_2.mhd"
    f1 = folder_ref / im_name
    f2 = folder_ff / im_name
    is_ok = (
        utility.assert_images(
            f1,
            f2,
            stats_ref,
            test_sad=False,
            sum_tolerance=70,
            scaleImageValuesFactor=scaling,
            axis="x",
        )
        and is_ok
    )

    o = folder_ff / "compare_2.pdf"
    plt = plot_spect_projection(f1, f2, scaling=10, islice=117, wslice=3)
    plt.savefig(o)
    print(f"Profile figure : ", o)

    utility.test_ok(is_ok)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
