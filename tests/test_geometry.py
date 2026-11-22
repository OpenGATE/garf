# -*- coding: utf-8 -*-
"""
Geometry: legacy angle convention, angle offset (shift to the crystal),
intersection of particles with the ARF plane (numpy and torch).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import garf
from conftest import legacy_angles, make_input_batch


def random_directions(n=1000, seed=1, uz_min=0.5):
    rng = np.random.default_rng(seed)
    uz = rng.uniform(uz_min, 1.0, n)
    a = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - uz * uz)
    return s * np.cos(a), s * np.sin(a), uz


def test_angle_offset_legacy_is_d_times_direction():
    """With legacy angles (theta=acos(uy), phi=acos(ux)), the shift is d*(ux, uy)."""
    ux, uy, _ = random_directions()
    theta, phi = legacy_angles(ux, uy)
    d = 74.625
    t = garf.compute_angle_offset_numpy(np.column_stack((theta, phi)), d)
    np.testing.assert_allclose(t[:, 0], d * ux, atol=1e-10)
    np.testing.assert_allclose(t[:, 1], d * uy, atol=1e-10)


def test_angle_offset_normal_incidence_is_zero():
    t = garf.compute_angle_offset_numpy(np.array([[90.0, 90.0]]), 74.625)
    np.testing.assert_allclose(t, 0, atol=1e-12)


def test_angle_offset_torch_equals_numpy():
    ux, uy, _ = random_directions()
    angles = np.column_stack(legacy_angles(ux, uy))
    tn = garf.compute_angle_offset_numpy(angles, 74.625)
    tt = garf.compute_angle_offset_torch(torch.from_numpy(angles), 74.625).numpy()
    np.testing.assert_allclose(tt, tn, atol=1e-10)


# ---------------------------------------------------------------------------
# plane intersection
# ---------------------------------------------------------------------------

RADIUS = 300.0
SIZE_MM = np.array([128 * 4.41806, 128 * 4.41806])


def source_batch(n=500, seed=3):
    """E, position (around the origin), direction towards +z: Ekine X Y Z dX dY dZ."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-50, 50, (n, 3))
    ux, uy, uz = random_directions(n, seed=seed, uz_min=0.8)
    E = rng.uniform(0.05, 0.16, n)
    return np.column_stack((E, pos, ux, uy, uz))


def expected_intersection(batch, rotation):
    """Reference: straight line to the plane at distance RADIUS, in the plane frame."""
    r_inv = rotation.inv()
    pos = r_inv.apply(batch[:, 1:4])
    d = r_inv.apply(batch[:, 4:7])
    s = (RADIUS - pos[:, 2]) / d[:, 2]
    keep = s > 0
    p = pos[keep] + s[keep, None] * d[keep]
    d = d[keep]
    inside = (np.abs(p[:, 0]) < SIZE_MM[0] / 2) & (np.abs(p[:, 1]) < SIZE_MM[1] / 2)
    theta, phi = legacy_angles(d[inside, 0], d[inside, 1])
    E = batch[keep][inside, 0]
    return np.column_stack((p[inside, 0], p[inside, 1], theta, phi, E))


@pytest.mark.parametrize("angle_deg", [0, 30, 90, 180])
def test_plane_intersection_numpy(angle_deg):
    rot = Rotation.from_euler("y", angle_deg, degrees=True)
    batch = source_batch()
    batch[:, 1:4] = rot.apply(batch[:, 1:4])
    batch[:, 4:7] = rot.apply(batch[:, 4:7])

    detector = SimpleNamespace(radius=RADIUS)
    plane = garf.garf_detector.GarfDetector.arf_plane_init_numpy(detector, rot, len(batch))
    projected = garf.arf_plane_intersection(batch, plane, SIZE_MM)

    expected = expected_intersection(batch, rot)
    np.testing.assert_allclose(projected, expected, atol=1e-8)


def torch_plane(rotation):
    detector = SimpleNamespace(
        current_gpu_mode="cpu",
        current_gpu_device=torch.device("cpu"),
        image_size_mm=torch.tensor(SIZE_MM),
    )
    plane = garf.GarfDetectorPlane(
        detector, np.array([0, 0, RADIUS]), rotation.as_matrix()
    )
    plane.initialize_torch()
    return plane


@pytest.mark.parametrize("angle_deg", [0, 30, 90, 180])
def test_plane_intersection_torch_equals_numpy(angle_deg):
    rot = Rotation.from_euler("y", angle_deg, degrees=True)
    batch = source_batch()
    batch[:, 1:4] = rot.apply(batch[:, 1:4])
    batch[:, 4:7] = rot.apply(batch[:, 4:7])

    plane = torch_plane(rot)
    projected = plane.plane_intersection(torch.from_numpy(batch).float()).numpy()

    expected = expected_intersection(batch, rot)
    assert projected.shape == expected.shape
    # float32 in the torch path; angles near 90 deg are sensitive (acos)
    np.testing.assert_allclose(projected[:, 0:2], expected[:, 0:2], atol=1e-3)
    np.testing.assert_allclose(projected[:, 2:4], expected[:, 2:4], atol=1e-2)
    np.testing.assert_allclose(projected[:, 4], expected[:, 4], rtol=1e-6)


def test_input_batch_directions_are_unit():
    b = make_input_batch()
    n = np.sqrt(b["ux"] ** 2 + b["uy"] ** 2 + b["uz"] ** 2)
    np.testing.assert_allclose(n, 1.0, atol=1e-12)
