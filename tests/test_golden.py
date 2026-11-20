# -*- coding: utf-8 -*-
"""
Golden tests: the current numbers for a fixed input batch and the legacy
model arf_Tc99m_v034.pth (legacy column layout x, y, theta, phi, E, w).
Any refactor must keep these, unless a change is intended and documented.
"""

import numpy as np

import garf
from conftest import DISTANCE_TO_CRYSTAL, IMAGE_SIZE, IMAGE_SPACING, legacy_columns

# network in float32: allow small differences between platforms / torch versions
RTOL = 1e-5
ATOL = 1e-8


def golden_batch(golden):
    return {k[3:]: v for k, v in golden.items() if k.startswith("in_")}


def test_golden_predict(golden, v034):
    _, model, model_data = v034
    px = legacy_columns(golden_batch(golden))
    probs = garf.nn_predict_numpy(model, model_data, px[:, 2:5].copy())
    np.testing.assert_allclose(probs, golden["probs"], rtol=RTOL, atol=ATOL)


def test_golden_points_to_image(golden, v034):
    _, model, model_data = v034
    px = legacy_columns(golden_batch(golden))
    u, v, w_pred = garf.arf_from_points_to_image_counts(
        px,
        model,
        model_data,
        DISTANCE_TO_CRYSTAL,
        IMAGE_SIZE * IMAGE_SPACING,
        IMAGE_SIZE,
        IMAGE_SPACING,
    )
    np.testing.assert_array_equal(u, golden["u"])
    np.testing.assert_array_equal(v, golden["v"])
    np.testing.assert_allclose(w_pred, golden["w_pred"], rtol=RTOL, atol=ATOL)

    image = np.zeros_like(golden["image"])
    garf.image_from_coordinates_add_numpy(image, u, v, w_pred, hit_slice=True)
    np.testing.assert_allclose(image, golden["image"], rtol=RTOL, atol=ATOL)


def test_golden_probabilities_are_normalised(golden):
    np.testing.assert_allclose(golden["probs"].sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(golden["probs"] >= 0)
