# -*- coding: utf-8 -*-
"""
Known bugs of garf <= 2.11, written as the *expected* behaviour and marked
xfail(strict=True): they fail today; when a bug is fixed the test passes,
strict mode turns that into an error, and the marker must be removed.

See docs/garf.md (versions and compatibility).
"""

import numpy as np
import pytest
import torch

import garf
from conftest import (
    DISTANCE_TO_CRYSTAL,
    IMAGE_SIZE,
    IMAGE_SPACING,
    gate_columns,
)
from test_image import torch_vs_numpy
from test_train import no_dataloader_worker, train_params  # noqa: F401 (fixture)


@pytest.mark.xfail(
    strict=True,
    reason="GATE >= 10.0.3 passes x, y, ux, uy, uz, E, w; garf reads x, y, theta, phi, E, w",
)
def test_gate_column_layout_gives_golden_result(golden, v034):
    _, model, model_data = v034
    b = {k[3:]: v for k, v in golden.items() if k.startswith("in_")}
    u, v, w_pred = garf.arf_from_points_to_image_counts(
        gate_columns(b),
        model,
        model_data,
        DISTANCE_TO_CRYSTAL,
        IMAGE_SIZE * IMAGE_SPACING,
        IMAGE_SIZE,
        IMAGE_SPACING,
    )
    np.testing.assert_array_equal(u, golden["u"])
    np.testing.assert_array_equal(v, golden["v"])
    np.testing.assert_allclose(w_pred, golden["w_pred"], rtol=1e-5, atol=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="torch path: x checked against size[1] and flat index y*shape[0]+x",
)
@pytest.mark.parametrize("hit_slice", [False, True])
def test_torch_image_equals_numpy_non_square(hit_slice):
    img_np, img_t = torch_vs_numpy((32, 16), (4.0, 4.0), hit_slice)
    # numpy image is (channels, ny, nx); the torch path allocates (channels, nx, ny)
    assert img_t.shape == img_np.shape
    np.testing.assert_allclose(img_t, img_np, rtol=1e-5, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="no shuffle + epoch stops at batch_per_epoch: only the first entries are used",
)
def test_training_uses_the_whole_dataset():
    # ordered dataset: first half class 0, second half class 1, same inputs
    n = 20_000
    rng = np.random.default_rng(2)
    x = np.column_stack(
        (rng.uniform(80, 100, n), rng.uniform(80, 100, n), np.full(n, 0.14))
    )
    x[:, 2] += rng.normal(0, 0.001, n)  # std must not be zero
    y = np.zeros(n)
    y[n // 2 :] = 1
    # 9 (+1) batches of 500 per epoch: 5000 entries, all in the first half
    params = train_params(
        H=8, L=1, epoch_max=5, batch_size=500, batch_per_epoch=9, learning_rate=1e-2
    )
    nn = garf.train_nn(x, y, params)
    model = garf.Net_v1(8, 1, nn["model_data"]["n_ene_win"])
    model.load_state_dict(nn["optim"]["model_state"][-1])
    md = dict(nn["model_data"], current_gpu_device=torch.device("cpu"))
    with torch.no_grad():
        probs = garf.nn_predict_numpy(model, md, x[:100])
    # the true probability of class 1 is 0.5 everywhere
    np.testing.assert_allclose(probs[:, 1], 0.5, atol=0.15)
