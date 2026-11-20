#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate tests/data/golden_v034.npz: reference outputs of the current garf
code for a fixed input batch and the legacy model arf_Tc99m_v034.pth.

Only rerun this when a change of the numbers is intended (and documented).
The inputs are stored in the file, so the golden data do not depend on the
numpy random generator version.

    python tests/generate_golden.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conftest import (  # noqa: E402
    GOLDEN_V034,
    DISTANCE_TO_CRYSTAL,
    IMAGE_SIZE,
    IMAGE_SPACING,
    load_v034,
    make_input_batch,
    legacy_columns,
)

import garf  # noqa: E402


def main():
    b = make_input_batch()
    _, model, model_data = load_v034()
    px = legacy_columns(b, with_weights=True)

    probs = garf.nn_predict_numpy(model, model_data, px[:, 2:5].copy())

    size_mm = IMAGE_SIZE * IMAGE_SPACING
    u, v, w_pred = garf.arf_from_points_to_image_counts(
        px,
        model,
        model_data,
        DISTANCE_TO_CRYSTAL,
        size_mm,
        IMAGE_SIZE,
        IMAGE_SPACING,
    )
    n_ene = model_data["n_ene_win"]
    image = np.zeros((n_ene, IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float64)
    garf.image_from_coordinates_add_numpy(image, u, v, w_pred, hit_slice=True)

    np.savez_compressed(
        GOLDEN_V034,
        **{f"in_{k}": val for k, val in b.items()},
        probs=probs,
        u=u,
        v=v,
        w_pred=w_pred,
        image=image,
    )
    print(f"Written {GOLDEN_V034}: {len(u)}/{len(px)} points in the image")
    print(f"image sums per channel: {image.sum(axis=(1, 2))}")


if __name__ == "__main__":
    main()
