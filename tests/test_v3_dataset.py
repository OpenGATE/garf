# -*- coding: utf-8 -*-
"""Training datasets: ROOT reading (v3 and old formats), metadata, split."""

import json

import numpy as np
import pytest
import uproot

import garf
from garf.dataset import TREE_NAME


def random_dataset(n=1000, seed=0, weight=False, **metadata):
    rng = np.random.default_rng(seed)
    uz = rng.uniform(0.2, 1, n)
    a = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1 - uz**2)
    w = rng.uniform(0.5, 2, n) if weight else None
    return garf.TrainingDataset(
        s * np.cos(a),
        s * np.sin(a),
        uz,
        rng.uniform(0.05, 0.16, n),
        rng.integers(0, 3, n),
        w,
        dict(metadata),
    )


@pytest.mark.parametrize("weight", [False, True])
def test_write_read_round_trip(tmp_path, weight):
    ds = random_dataset(weight=weight, rr=50, n_channels=3)
    f = tmp_path / "train.root"
    garf.write_training_dataset(f, ds)
    r = garf.load_training_dataset(f)
    for k in ("ux", "uy", "uz", "E", "window"):
        np.testing.assert_array_equal(getattr(r, k), getattr(ds, k))
    if weight:
        np.testing.assert_array_equal(r.weight, ds.weight)
    else:
        assert r.weight is None
    assert r.rr == 50 and r.n_channels == 3
    assert r.metadata["format"] == "directions"


def write_old(path, theta, phi, E, window, window_key="window"):
    with uproot.recreate(path) as f:
        f[TREE_NAME] = {"E": E, "Theta": theta, "Phi": phi, window_key: window}


@pytest.mark.parametrize("convention", ["legacy_acos", "theta_phi"])
@pytest.mark.parametrize("window_key", ["window", "w"])
def test_old_dataset(tmp_path, convention, window_key):
    ds = random_dataset()
    feats = garf.compute_features(
        convention,
        *[__import__("torch").from_numpy(a) for a in (ds.ux, ds.uy, ds.uz, ds.E)],
    ).numpy()
    f = tmp_path / "old.root"
    write_old(f, feats[:, 0], feats[:, 1], ds.E, ds.window.astype(float), window_key)
    with pytest.raises(ValueError, match="angle convention must be given"):
        garf.load_training_dataset(f)
    r = garf.load_training_dataset(f, angle_convention=convention)
    np.testing.assert_allclose(r.ux, ds.ux, atol=1e-9)
    np.testing.assert_allclose(r.uy, ds.uy, atol=1e-9)
    np.testing.assert_allclose(r.uz, ds.uz, atol=1e-6)
    np.testing.assert_array_equal(r.window, ds.window)
    assert r.metadata["angle_convention"] == convention


def test_old_dataset_convention_from_sidecar(tmp_path):
    """The GATE training actor writes the convention in <name>.json."""
    import torch

    ds = random_dataset()
    t = [torch.from_numpy(a) for a in (ds.ux, ds.uy, ds.uz, ds.E)]
    feats = garf.compute_features("theta_phi", *t).numpy()
    f = tmp_path / "old.root"
    write_old(f, feats[:, 0], feats[:, 1], ds.E, ds.window.astype(float))
    f.with_suffix(".json").write_text(
        json.dumps({"angle_convention": "theta_phi", "rr": 20})
    )
    r = garf.load_training_dataset(f)
    np.testing.assert_allclose(r.ux, ds.ux, atol=1e-9)
    assert r.rr == 20 and r.metadata["angle_convention"] == "theta_phi"
    # an explicit argument wins (and is checked against the angles)
    with pytest.raises(ValueError, match="theta_phi dataset"):
        garf.load_training_dataset(f, angle_convention="legacy_acos")


def test_old_dataset_wrong_convention(tmp_path):
    ds = random_dataset()
    import torch

    t = [torch.from_numpy(a) for a in (ds.ux, ds.uy, ds.uz, ds.E)]
    feats = garf.compute_features("theta_phi", *t).numpy()
    f = tmp_path / "old.root"
    write_old(f, feats[:, 0], feats[:, 1], ds.E, ds.window.astype(float))
    with pytest.raises(ValueError, match="theta_phi dataset"):
        garf.load_training_dataset(f, angle_convention="legacy_acos")


def test_not_a_training_dataset(tmp_path):
    f = tmp_path / "x.root"
    with uproot.recreate(f) as r:
        r["other"] = {"a": np.zeros(3)}
    with pytest.raises(ValueError, match="no tree"):
        garf.load_training_dataset(f)
    with pytest.raises(FileNotFoundError):
        garf.load_training_dataset(tmp_path / "missing.root")


def test_n_channels():
    ds = random_dataset()
    assert ds.n_channels == 3
    ds.metadata["n_channels"] = 4  # a window with no entry
    assert ds.n_channels == 4
    ds.metadata["n_channels"] = 2
    with pytest.raises(ValueError, match="n_channels"):
        ds.n_channels


def test_split():
    ds = random_dataset(n=1000, weight=True)
    tr, va = ds.split(0.2, seed=1)
    assert len(tr) == 800 and len(va) == 200
    both = np.sort(np.concatenate((tr.E, va.E)))
    np.testing.assert_array_equal(both, np.sort(ds.E))
    tr2, _ = ds.split(0.2, seed=1)
    np.testing.assert_array_equal(tr.E, tr2.E)
    tr0, va0 = ds.split(0, seed=1)
    assert len(tr0) == 1000 and len(va0) == 0


def test_sidecar_metadata(tmp_path):
    ds = random_dataset(rr=20, detector={"distance_to_crystal": 74.625})
    f = tmp_path / "t.root"
    garf.write_training_dataset(f, ds)
    meta = json.loads(f.with_suffix(".json").read_text())
    assert meta["rr"] == 20 and meta["detector"]["distance_to_crystal"] == 74.625
    assert garf.load_training_dataset(f).metadata["detector"] == meta["detector"]
