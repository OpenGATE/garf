# -*- coding: utf-8 -*-
"""
From points on the plane to image counts: network output normalisation,
Russian roulette, mm -> pixel, out-of-image cut, accumulation (numpy, torch).

A constant "network" (fixed probabilities) makes the expected values exact.

Conventions (as in garf <= 2.11 numpy path):
- image size and spacing are given as (nx, ny)
- the image array is (channels, ny, nx), filled as img[c, iy, ix]
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import garf

DEV = torch.device("cpu")


class ConstantModel(torch.nn.Module):
    """Returns log(p) whatever the input: softmax gives back p."""

    def __init__(self, p):
        super().__init__()
        self.logp = torch.log(torch.tensor(p, dtype=torch.float32))

    def forward(self, x):
        return self.logp.repeat(x.shape[0], 1)


def constant_model(p, rr=1):
    model_data = dict(
        x_mean=np.zeros(3), x_std=np.ones(3), RR=rr, current_gpu_device=DEV
    )
    return ConstantModel(p), model_data


def expected_probs(p, rr):
    p = np.array(p, dtype=np.float64)
    p[0] *= rr
    return p / p.sum()


@pytest.mark.parametrize("rr", [1, 50])
def test_predict_russian_roulette_normalisation(rr):
    p = [0.9, 0.04, 0.06]
    model, model_data = constant_model(p, rr)
    probs = garf.nn_predict_numpy(model, model_data, np.zeros((10, 3)))
    np.testing.assert_allclose(probs, [expected_probs(p, rr)] * 10, rtol=1e-6)


def test_predict_accepts_lower_case_rr_key():
    p = [0.9, 0.04, 0.06]
    model, model_data = constant_model(p, 50)
    model_data["rr"] = model_data.pop("RR")
    probs = garf.nn_predict_numpy(model, model_data, np.zeros((2, 3)))
    np.testing.assert_allclose(probs[0], expected_probs(p, 50), rtol=1e-6)


def pixel_centres(ix, iy, size, spacing):
    """mm coordinates of the centre of pixel (ix, iy); the image is centred on 0."""
    x = (ix + 0.5) * spacing[0] - size[0] * spacing[0] / 2
    y = (iy + 0.5) * spacing[1] - size[1] * spacing[1] / 2
    return x, y


def normal_incidence_batch(x, y, E=0.14, w=None):
    """Legacy layout x, y, theta, phi, E [, w], normal incidence (theta=phi=90)."""
    n = len(x)
    cols = [x, y, np.full(n, 90.0), np.full(n, 90.0), np.full(n, E)]
    if w is not None:
        cols.append(w)
    return np.column_stack(cols)


SIZES = [((128, 128), (4.41806, 4.41806)), ((64, 32), (2.0, 3.0))]


@pytest.mark.parametrize("size, spacing", SIZES)
def test_pixel_centres_map_to_their_pixel(size, spacing):
    size, spacing = np.array(size), np.array(spacing)
    rng = np.random.default_rng(7)
    ix = rng.integers(0, size[0], 200)
    iy = rng.integers(0, size[1], 200)
    x, y = pixel_centres(ix, iy, size, spacing)
    model, model_data = constant_model([0.5, 0.2, 0.3])
    u, v, _ = garf.arf_from_points_to_image_counts(
        normal_incidence_batch(x, y),
        model,
        model_data,
        74.625,
        size * spacing,
        size,
        spacing,
    )
    np.testing.assert_array_equal(v, ix)  # v is the x index
    np.testing.assert_array_equal(u, iy)  # u is the y index


@pytest.mark.parametrize("size, spacing", SIZES)
def test_points_outside_are_removed(size, spacing):
    size, spacing = np.array(size), np.array(spacing)
    half = size * spacing / 2
    eps = 0.01 * spacing
    x = np.array([0, -half[0] - eps[0], half[0] + eps[0], 0, 0, -half[0] + eps[0]])
    y = np.array([0, 0, 0, -half[1] - eps[1], half[1] + eps[1], half[1] - eps[1]])
    model, model_data = constant_model([0.5, 0.2, 0.3])
    u, v, w_pred = garf.arf_from_points_to_image_counts(
        normal_incidence_batch(x, y), model, model_data, 0.0, 2 * half, size, spacing
    )
    # kept: centre, and the point just inside the (-x, +y) corner
    assert len(u) == len(v) == len(w_pred) == 2
    np.testing.assert_array_equal(v, [size[0] // 2, 0])
    np.testing.assert_array_equal(u, [size[1] // 2, size[1] - 1])


def test_particle_weights_multiply_probabilities():
    p, rr = [0.9, 0.04, 0.06], 50
    model, model_data = constant_model(p, rr)
    w = np.array([1.0, 2.0, 0.5])
    size, spacing = np.array([16, 16]), np.array([1.0, 1.0])
    _, _, w_pred = garf.arf_from_points_to_image_counts(
        normal_incidence_batch(np.zeros(3), np.zeros(3), w=w),
        model,
        model_data,
        0.0,
        size * spacing,
        size,
        spacing,
    )
    np.testing.assert_allclose(w_pred, np.outer(w, expected_probs(p, rr)), rtol=1e-6)


@pytest.mark.parametrize("hit_slice", [False, True])
@pytest.mark.parametrize("size", [(16, 16), (16, 8)])
def test_image_accumulation_numpy(hit_slice, size):
    nx, ny = size
    rng = np.random.default_rng(11)
    n = 500
    v = rng.integers(0, nx, n)  # x index
    u = rng.integers(0, ny, n)  # y index
    w_pred = rng.uniform(0, 1, (n, 3))

    n_channels = 3 if hit_slice else 2
    img = np.zeros((n_channels, ny, nx))
    garf.image_from_coordinates_add_numpy(img, u, v, w_pred, hit_slice=hit_slice)

    expected = np.zeros_like(img)
    first = 1 if hit_slice else 0
    for k in range(n):
        for c in range(1, 3):
            expected[c - 1 + first, u[k], v[k]] += w_pred[k, c]
        if hit_slice:
            expected[0, u[k], v[k]] += 1
    np.testing.assert_allclose(img, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# torch path of GarfDetector (used by gaga_phsp)
# ---------------------------------------------------------------------------


def torch_detector(model, size, spacing, rr, n_ene, distance, hit_slice):
    """A GarfDetector with the members used by build_image_from_projected_points_torch."""
    d = garf.GarfDetector()
    d.model = model
    d.current_gpu_mode = "cpu"
    d.current_gpu_device = DEV
    d.x_mean = torch.zeros(3, dtype=torch.float64)
    d.x_std = torch.ones(3, dtype=torch.float64)
    d.rr = rr
    d.nb_ene = n_ene
    d.crystal_distance = distance
    d.hit_slice_flag = hit_slice
    d.image_size = torch.tensor(size)
    d.image_spacing = torch.tensor(spacing, dtype=torch.float64)
    d.image_size_mm = d.image_size * d.image_spacing
    n_channels = n_ene if hit_slice else n_ene - 1
    # torch path allocates [projections, channels, image_size[0], image_size[1]]
    d.output_image = torch.zeros((1, n_channels, size[0], size[1]))
    return d


def torch_vs_numpy(size, spacing, hit_slice):
    size, spacing = np.array(size), np.array(spacing)
    rng = np.random.default_rng(5)
    n = 1000
    half = size * spacing / 2
    x = rng.uniform(-half[0], half[0], n)
    y = rng.uniform(-half[1], half[1], n)
    theta = rng.uniform(80, 100, n)
    phi = rng.uniform(80, 100, n)
    batch = np.column_stack((x, y, theta, phi, np.full(n, 0.14)))

    p, rr, d = [0.9, 0.04, 0.06], 50, 10.0
    model, model_data = constant_model(p, rr)
    u, v, w_pred = garf.arf_from_points_to_image_counts(
        batch, model, model_data, d, size * spacing, size, spacing
    )
    n_channels = 3 if hit_slice else 2
    img_np = np.zeros((n_channels, size[1], size[0]))
    garf.image_from_coordinates_add_numpy(img_np, u, v, w_pred, hit_slice=hit_slice)

    det = torch_detector(model, size, spacing, rr, 3, d, hit_slice)
    with torch.no_grad():
        det.build_image_from_projected_points_torch(torch.from_numpy(batch), 0)
    img_t = det.output_image[0].numpy()
    return img_np, img_t


@pytest.mark.parametrize("hit_slice", [False, True])
def test_torch_image_equals_numpy_square(hit_slice):
    img_np, img_t = torch_vs_numpy((32, 32), (4.0, 4.0), hit_slice)
    np.testing.assert_allclose(img_t, img_np, rtol=1e-5, atol=1e-6)
