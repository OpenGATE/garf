# -*- coding: utf-8 -*-
"""
garf v3 API (ArfModel, ArfProjector) against the golden data of garf 2.11:
the legacy model, legacy features and legacy shift must give the same numbers,
now from the physical quantities passed by GATE (x, y, ux, uy, uz, E, w).
"""

import os
import warnings

import numpy as np
import pytest
import torch

import garf
from conftest import DISTANCE_TO_CRYSTAL, IMAGE_SIZE, IMAGE_SPACING, PTH_V034

RTOL = 1e-5
ATOL = 1e-8


def load_legacy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return garf.ArfModel.load(PTH_V034, device="cpu")


@pytest.fixture(scope="module")
def legacy_model():
    return load_legacy()


@pytest.fixture(scope="module")
def converted_model(tmp_path_factory, legacy_model):
    path = tmp_path_factory.mktemp("garf") / "v034.garf"
    legacy_model.save(path)
    return garf.ArfModel.load(path, device="cpu")


@pytest.fixture(params=["pth", "garf"])
def model(request, legacy_model, converted_model):
    return legacy_model if request.param == "pth" else converted_model


def batch(golden):
    return {k[3:]: v for k, v in golden.items() if k.startswith("in_")}


def projector(**kw):
    return garf.ArfProjector(
        IMAGE_SIZE, IMAGE_SPACING, distance_to_crystal=DISTANCE_TO_CRYSTAL, **kw
    )


def test_legacy_pth_metadata(legacy_model):
    m = legacy_model
    assert m.features == "legacy_acos"
    assert m.n_channels == 3 and m.n_windows == 2
    assert m.rr == 50
    assert m.architecture == dict(
        n_in=3, H=200, L=3, n_out=3, activation="relu", block="mlp"
    )
    assert m.metadata["training"]["stored_epochs"] == [0, 20]
    assert m.metadata["training"]["selected_epoch"] == 20


def test_legacy_pth_warns():
    with pytest.warns(UserWarning, match="legacy .pth"):
        garf.ArfModel.load(PTH_V034, device="cpu")


def test_predict_golden(model, golden):
    b = batch(golden)
    probs = model.predict(b["ux"], b["uy"], b["uz"], b["E"])
    assert isinstance(probs, np.ndarray) and probs.dtype == np.float64
    np.testing.assert_allclose(probs, golden["probs"], rtol=RTOL, atol=ATOL)


def test_predict_torch_input(model, golden):
    b = batch(golden)
    t = {k: torch.from_numpy(b[k]) for k in ("ux", "uy", "uz", "E")}
    probs = model.predict(t["ux"], t["uy"], t["uz"], t["E"])
    assert isinstance(probs, torch.Tensor)
    np.testing.assert_allclose(probs.numpy(), golden["probs"], rtol=RTOL, atol=1e-7)


def test_project_golden(model, golden):
    b = batch(golden)
    ix, iy, values = projector().project(
        model, b["x"], b["y"], b["ux"], b["uy"], b["uz"], b["E"], b["w"]
    )
    np.testing.assert_array_equal(ix, golden["v"])
    np.testing.assert_array_equal(iy, golden["u"])
    np.testing.assert_allclose(values, golden["w_pred"][:, 1:], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("hit_slice", [True, False])
def test_accumulate_golden(model, golden, hit_slice):
    b = batch(golden)
    p = projector(hit_slice=hit_slice)
    image = np.zeros(p.image_shape(model))
    squared = np.zeros_like(image)
    n = p.accumulate(
        image,
        model,
        b["x"],
        b["y"],
        b["ux"],
        b["uy"],
        b["uz"],
        b["E"],
        b["w"],
        squared_image=squared,
    )
    assert n == len(golden["u"])
    expected = golden["image"] if hit_slice else golden["image"][1:]
    np.testing.assert_allclose(image, expected, rtol=RTOL, atol=ATOL)
    # squared: sum of (w p)^2 per pixel
    expected_sq = np.zeros_like(image)
    values = golden["w_pred"][:, 1:]
    first = 1 if hit_slice else 0
    for c in range(values.shape[1]):
        np.add.at(expected_sq[c + first], (golden["u"], golden["v"]), values[:, c] ** 2)
    if hit_slice:
        np.add.at(expected_sq[0], (golden["u"], golden["v"]), 1.0)
    np.testing.assert_allclose(squared, expected_sq, rtol=RTOL, atol=ATOL)


def test_accumulate_torch_image(model, golden):
    """torch inputs and image (as in GarfDetector), float32."""
    b = batch(golden)
    p = projector(hit_slice=True)
    image = torch.zeros(p.image_shape(model), dtype=torch.float32)
    t = {k: torch.from_numpy(v).float() for k, v in b.items()}
    p.accumulate(
        image, model, t["x"], t["y"], t["ux"], t["uy"], t["uz"], t["E"], t["w"]
    )
    # float32 geometry: a few points may change pixel; compare the totals
    np.testing.assert_allclose(
        image.sum(dim=(1, 2)).numpy(), golden["image"].sum(axis=(1, 2)), rtol=1e-4
    )


def test_save_load_round_trip(legacy_model, converted_model, tmp_path):
    m = converted_model
    assert m.features == legacy_model.features
    assert m.rr == legacy_model.rr
    assert m.architecture == legacy_model.architecture
    assert m.metadata["training"]["selected_epoch"] == 20
    # second round trip is stable
    path = tmp_path / "again.garf"
    m.save(path)
    m2 = garf.ArfModel.load(path, device="cpu")
    for a, b in zip(m.nets[0].state_dict().values(), m2.nets[0].state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_garf_file_is_not_pickle(converted_model, tmp_path):
    path = tmp_path / "m.garf"
    converted_model.save(path)
    meta = garf.model.read_garf_metadata(path)
    assert meta["format"] == "garf-arf" and meta["format_version"] == 1
    assert meta["model_type"] == "mlp" and meta["features"] == "legacy_acos"
    # one state only: ~122k float32 parameters (the .pth stores two states, 983 kB)
    assert path.stat().st_size < 0.6 * PTH_V034.stat().st_size


def test_wrong_format_version(converted_model, tmp_path):
    from safetensors.torch import load_file, save_file
    import json

    path = tmp_path / "m.garf"
    converted_model.save(path)
    meta = garf.model.read_garf_metadata(path)
    meta["format_version"] = 99
    save_file(load_file(str(path)), str(path), metadata={"garf": json.dumps(meta)})
    with pytest.raises(ValueError, match="format version 99"):
        garf.ArfModel.load(path, device="cpu")


def test_ensemble_averages_probabilities(legacy_model):
    m = legacy_model
    torch.manual_seed(3)
    other = garf.MlpArfModel.build_net(m.architecture)
    ens = garf.MlpArfModel(
        [m.nets[0], other],
        m.x_mean,
        m.x_std,
        m.features,
        m.n_channels,
        m.rr,
        m.architecture,
    )
    single = garf.MlpArfModel(
        [other], m.x_mean, m.x_std, m.features, m.n_channels, m.rr, m.architecture
    )
    rng = np.random.default_rng(0)
    uz = rng.uniform(0.9, 1, 50)
    ux = np.sqrt(1 - uz**2)
    uy = np.zeros(50)
    E = np.full(50, 0.14)
    expected = (m.predict(ux, uy, uz, E) + single.predict(ux, uy, uz, E)) / 2
    np.testing.assert_allclose(ens.predict(ux, uy, uz, E), expected, rtol=1e-6)


def test_projector_distance_from_model(converted_model, golden, tmp_path):
    m = converted_model
    m.metadata["detector"] = {"distance_to_crystal": DISTANCE_TO_CRYSTAL}
    b = batch(golden)
    p = garf.ArfProjector(IMAGE_SIZE, IMAGE_SPACING)
    ix, iy, _ = p.project(m, b["x"], b["y"], b["ux"], b["uy"], b["uz"], b["E"])
    np.testing.assert_array_equal(ix, golden["v"])
    with pytest.warns(UserWarning, match="differs"):
        garf.ArfProjector(IMAGE_SIZE, IMAGE_SPACING, distance_to_crystal=70).pixels(
            b["x"], b["y"], b["ux"], b["uy"], b["uz"], model=m
        )
    del m.metadata["detector"]


def test_projector_needs_a_distance(converted_model):
    p = garf.ArfProjector(IMAGE_SIZE, IMAGE_SPACING)
    with pytest.raises(ValueError, match="distance_to_crystal"):
        p.pixels(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            converted_model,
        )


def test_shift_modes():
    d = 50.0
    uz = np.array([0.8, 0.6])
    ux = np.array([0.6, 0.0])
    uy = np.array([0.0, 0.8])
    zero = np.zeros(2)
    size, sp = (2000, 2000), (0.1, 0.1)  # 200 mm, fine pixels
    for mode, scale in [("legacy", 1.0), ("straight", 1 / uz)]:
        p = garf.ArfProjector(size, sp, distance_to_crystal=d, shift_mode=mode)
        ix, iy, inside = p.pixels(zero, zero, ux, uy, uz)
        assert inside.all()
        np.testing.assert_allclose((ix + 0.5) * sp[0] - 100, d * ux * scale, atol=sp[0])
        np.testing.assert_allclose((iy + 0.5) * sp[1] - 100, d * uy * scale, atol=sp[1])


def test_non_square_image(legacy_model):
    """(nx, ny) = (32, 16): image (channels, 16, 32), img[c, iy, ix]."""
    p = garf.ArfProjector((32, 16), (2.0, 3.0), distance_to_crystal=0, hit_slice=True)
    x = np.array([-31.0, 31.0, 0.5])
    y = np.array([-23.0, 23.0, 0.5])
    one = np.ones(3)
    image = np.zeros(p.image_shape(legacy_model))
    assert image.shape == (3, 16, 32)
    p.accumulate(image, legacy_model, x, y, 0 * one, 0 * one, one, 0.14 * one)
    assert image[0, 0, 0] == 1 and image[0, 15, 31] == 1 and image[0, 8, 16] == 1
    assert image[0].sum() == 3


def test_wrong_image_shape(legacy_model):
    p = projector()
    with pytest.raises(ValueError, match="image shape"):
        p.accumulate(
            np.zeros((2, 64, 64)),
            legacy_model,
            *[np.zeros(1)] * 4,
            np.ones(1),
            np.ones(1),
        )


def test_inputs_must_have_the_same_length(legacy_model):
    with pytest.raises(ValueError, match="same length"):
        legacy_model.predict(np.zeros(3), np.zeros(3), np.ones(2), np.ones(3))


def two_state_pth(tmp_path):
    """A legacy .pth with two stored states (seeds 1 and 2)."""
    states = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        states.append(garf.Net_v1(4, 1, 3).state_dict())
    md = dict(H=4, L=1, n_ene_win=3, RR=1, x_mean=np.zeros(3), x_std=np.ones(3))
    data = [dict(epoch=10, train_loss=0.1), dict(epoch=20, train_loss=0.2)]
    pth = tmp_path / "two_states.pth"
    torch.save(dict(model_data=md, optim=dict(model_state=states, data=data)), pth)
    return pth, states


@pytest.mark.parametrize("index", [-1, 0])
def test_legacy_state_selection(tmp_path, index):
    """
    Default: the last stored state, as garf <= 2.11 always used. The
    best_epoch_index key is not reliable (an index in old garf versions, an
    epoch number in garf 2.11), so it is not used.
    """
    pth, states = two_state_pth(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = garf.ArfModel.load(pth, device="cpu", legacy_state_index=index)
    for k, v in states[index].items():
        torch.testing.assert_close(m.nets[0].state_dict()[k], v)
    assert m.metadata["training"]["selected_epoch"] == [10, 20][index]


def test_garf_convert_cli(tmp_path, golden):
    from click.testing import CliRunner
    from garf.bin.garf_convert import garf_convert

    out = tmp_path / "v034.garf"
    r = CliRunner().invoke(
        garf_convert, [str(PTH_V034), str(out), "-d", str(DISTANCE_TO_CRYSTAL)]
    )
    assert r.exit_code == 0, r.output
    m = garf.ArfModel.load(out, device="cpu")
    assert m.distance_to_crystal == DISTANCE_TO_CRYSTAL
    b = batch(golden)
    ix, iy, values = garf.ArfProjector(IMAGE_SIZE, IMAGE_SPACING).project(
        m, b["x"], b["y"], b["ux"], b["uy"], b["uz"], b["E"], b["w"]
    )
    np.testing.assert_array_equal(ix, golden["v"])
    np.testing.assert_allclose(values, golden["w_pred"][:, 1:], rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or "GITHUB_ACTIONS" in os.environ,
    reason="needs a usable mps device (detected but not usable on GitHub macOS runners)",
)
def test_projector_mps_equals_cpu(v034_model_cpu_and_mps):
    """A non-contiguous mps -> cpu copy gave garbage images (found with GATE)."""
    m_cpu, m_mps = v034_model_cpu_and_mps
    rng = np.random.default_rng(3)
    n = 20_000
    u = rng.normal(size=(n, 3))
    u[:, 2] = np.abs(u[:, 2]) + 0.5
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x, y = rng.uniform(-200, 200, (2, n))
    E = rng.uniform(0.07, 0.15, n)
    w = rng.uniform(0.5, 2, n)
    for hit_slice in (False, True):
        pr = garf.ArfProjector((64, 64), (8.0, 8.0), 74.625, hit_slice=hit_slice)
        images = []
        for m in (m_cpu, m_mps):
            img = np.zeros(pr.image_shape(m))
            pr.accumulate(img, m, x, y, *u.T, E, w)
            images.append(img)
        np.testing.assert_allclose(images[1], images[0], rtol=1e-4, atol=1e-9)
        _, _, v = pr.project(m_mps, x, y, *u.T, E, w)
        _, _, v0 = pr.project(m_cpu, x, y, *u.T, E, w)
        np.testing.assert_allclose(v, v0, rtol=1e-4, atol=1e-9)


@pytest.fixture
def v034_model_cpu_and_mps():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            garf.ArfModel.load(PTH_V034, device="cpu"),
            garf.ArfModel.load(PTH_V034, device="mps"),
        )
