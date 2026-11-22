# -*- coding: utf-8 -*-
"""
v3 GarfDetector (standalone, used by gaga-phsp) against the garf 2.11
GarfDetector (numpy and torch paths), on the same particles.
"""

import os
import warnings

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import garf
import garf.garf_detector as old
from conftest import DISTANCE_TO_CRYSTAL, PTH_V034

RADIUS = 300.0
SIZE = (64, 64)
SPACING = (4.8, 4.8)
# as in gaga-phsp (Intevo): plane rotation like the detector, then head-foot and gantry
INITIAL = Rotation.from_euler("xz", (180, 90), degrees=True) * Rotation.from_euler(
    "yx", (90, 90), degrees=True
)
GANTRY = [Rotation.from_euler("z", a, degrees=True) for a in (0, 100, 230)]


def particles(n=50_000, seed=4):
    """E, x, y, z, ux, uy, uz: a 50 mm cube around the origin, isotropic."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-50, 50, (n, 3))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    E = np.where(rng.uniform(0, 1, n) < 0.7, 0.1405, rng.uniform(0.05, 0.16, n))
    return np.column_stack((E, pos, u))


def new_detector(**kw):
    d = garf.GarfDetector()
    d.model = load_model()
    d.device = "cpu"
    d.image_size = SIZE
    d.image_spacing = SPACING
    d.distance_to_crystal = DISTANCE_TO_CRYSTAL
    d.radius = RADIUS
    d.initial_plane_rotation = INITIAL
    for k, v in kw.items():
        setattr(d, k, v)
    d.initialize(GANTRY)
    return d


def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return garf.ArfModel.load(PTH_V034, device="cpu")


def old_detector(hit_slice):
    d = old.GarfDetector()
    d.pth_filename = PTH_V034
    d.gpu_mode = "cpu"
    d.batch_size = 1e6
    d.image_size = np.array(SIZE)
    d.image_spacing = np.array(SPACING)
    d.crystal_distance = DISTANCE_TO_CRYSTAL
    d.radius = RADIUS
    d.initial_plane_rotation = INITIAL
    d.hit_slice_flag = hit_slice
    d.initialize(GANTRY)
    d.model_data["current_gpu_device"] = torch.device("cpu")
    return d


def old_numpy_images(batch, hit_slice):
    """
    The loop of gaga-phsp generate_projections_numpy, with the 2.11 detector.
    Always run with the hit slice, then drop it if not requested: without the
    hit slice, image_from_coordinates_add_numpy writes the windows in channels
    0..n-2 while gaga-phsp drops channel 0 (window 1 lost, window 2 in the
    first channel, second channel empty): a bug of that combination.
    """
    d = old_detector(True)
    n = len(GANTRY)
    planes = d.initialize_planes_numpy(len(batch))
    projected = [None] * n
    img = np.zeros((n, d.nb_ene, SIZE[1], SIZE[0]))
    for i in range(n):
        d.project_to_planes_numpy(batch, i, planes, projected, img)
    for i in range(n):
        if projected[i] is not None and len(projected[i]):
            d.build_image_from_projected_points_numpy(projected[i], img[i])
    return img if hit_slice else img[:, 1:]


def old_torch_images(batch, hit_slice):
    d = old_detector(hit_slice)
    d.initialize_torch()
    with torch.no_grad():
        d.project_to_planes_torch(torch.from_numpy(batch).float(), [None] * len(GANTRY))
    return d.output_image.numpy()


@pytest.mark.parametrize("hit_slice", [False, True])
def test_detector_equals_old_numpy_path(hit_slice):
    batch = particles()
    expected = old_numpy_images(batch, hit_slice)
    d = new_detector(hit_slice=hit_slice)
    n = d.add(batch)
    img = d.image_arrays.numpy()
    assert img.shape == expected.shape
    assert expected.sum() > 0 and n > 1000
    np.testing.assert_allclose(img, expected, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("hit_slice", [False, True])
def test_detector_close_to_old_torch_path(hit_slice):
    """The 2.11 torch path is float32: compare the totals per angle and channel."""
    batch = particles()
    expected = old_torch_images(batch, hit_slice)
    d = new_detector(hit_slice=hit_slice)
    d.add(batch)
    img = d.image_arrays.numpy()
    assert img.shape == expected.shape
    np.testing.assert_allclose(
        img.sum(axis=(2, 3)), expected.sum(axis=(2, 3)), rtol=1e-3
    )


def test_numpy_and_torch_batches_and_chunks():
    batch = particles(20_000)
    d1 = new_detector()
    d1.add(batch)
    d2 = new_detector(batch_size=3000)  # several chunks
    d2.add(torch.from_numpy(batch))
    np.testing.assert_allclose(
        d2.image_arrays.numpy(), d1.image_arrays.numpy(), rtol=1e-12
    )


def test_non_square_image():
    """Was broken in the 2.11 torch path."""
    batch = particles(20_000)
    d = new_detector(image_size=(64, 32), image_spacing=(4.8, 4.8))
    d.add(batch)
    assert d.image_arrays.shape == (3, 2, 32, 64)
    assert d.image_arrays.sum() > 0


def test_images_itk():
    d = new_detector(hit_slice=True)
    d.add(particles(10_000))
    images = d.images()
    assert len(images) == 3
    img = images[0]
    assert tuple(img.GetLargestPossibleRegion().GetSize()) == (64, 64, 3)
    np.testing.assert_allclose(img.GetSpacing(), (4.8, 4.8, 1))
    np.testing.assert_allclose(img.GetOrigin(), (-64 * 2.4 + 2.4, -64 * 2.4 + 2.4, 0))


def test_wrong_batch_shape():
    d = new_detector()
    with pytest.raises(ValueError, match=r"\(n, 7\)"):
        d.add(np.zeros((10, 6)))


def test_missing_parameter():
    d = garf.GarfDetector()
    d.model = load_model()
    with pytest.raises(ValueError, match="image_size"):
        d.initialize(GANTRY)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or "GITHUB_ACTIONS" in os.environ,
    reason="needs a usable mps device (detected but not usable on GitHub macOS runners)",
)
def test_detector_mps_equals_cpu():
    batch = particles(20_000)
    d_cpu = new_detector(hit_slice=True)
    d_cpu.add(batch)
    d_mps = new_detector(hit_slice=True, model=load_model(), device="mps")
    d_mps.add(batch)
    np.testing.assert_allclose(
        d_mps.image_arrays.cpu().numpy(),
        d_cpu.image_arrays.numpy(),
        rtol=1e-3,
        atol=1e-6,
    )
