# -*- coding: utf-8 -*-
"""
v3 trainer on synthetic data with a known detection probability (the same
problem as test_train.py, for the garf 2.11 trainer).
"""

import json

import numpy as np
import pytest

import garf
from test_train import synthetic_dataset, true_probs

FAST = dict(H=32, L=1, epochs=40, learning_rate=1e-2, batch_size=1000, device="cpu")


def dataset(n=100_000, seed=1, **metadata):
    x, y = synthetic_dataset(n, seed)
    ux, uy, uz = garf.direction_from_angles("legacy_acos", x[:, 0], x[:, 1])
    return garf.TrainingDataset(ux, uy, uz, x[:, 2], y, metadata=dict(rr=1, **metadata))


def errors(model):
    g = np.linspace(75, 105, 13)
    th, ph = [a.ravel() for a in np.meshgrid(g, g)]
    ux, uy, uz = garf.direction_from_angles("legacy_acos", th, ph)
    e = []
    for E in [0.1405, 0.09]:
        Ev = np.full(len(th), E)
        e.append(np.abs(model.predict(ux, uy, uz, Ev) - true_probs(th, ph, Ev)))
    return np.concatenate(e)


@pytest.mark.parametrize("features", ["ux_uy_E", "legacy_acos", "theta_phi"])
def test_training_recovers_known_probabilities(features):
    m = garf.train(
        dataset(), features=features, seed=1234, early_stopping=0, verbose=False, **FAST
    )
    e = errors(m)
    # 2026-11-20: mean 0.009-0.010, max 0.06-0.07 (ux_uy_E, legacy_acos);
    # garf 2.11 trainer on the same problem: mean 0.014, max 0.091
    assert e.mean() < 0.015
    assert e.max() < (0.12 if features == "theta_phi" else 0.09)
    assert m.features == features
    h = m.metadata["training"]["networks"][0]
    assert h["selection"] == "validation"
    assert h["selected_loss"] == min(h["validation_loss"])


@pytest.mark.filterwarnings("ignore:garf\\: an input feature is constant")
def test_training_uses_the_whole_dataset():
    """Ordered dataset (first half channel 0, second half channel 1): see test_known_bugs."""
    n = 20_000
    rng = np.random.default_rng(2)
    uz = rng.uniform(0.9, 1, n)
    ux = np.sqrt(1 - uz**2)
    window = np.zeros(n, dtype=int)
    window[n // 2 :] = 1
    ds = garf.TrainingDataset(
        ux, 0 * ux, uz, rng.uniform(0.1, 0.15, n), window, metadata={"rr": 1}
    )
    m = garf.train(
        ds,
        H=8,
        L=1,
        epochs=5,
        batch_size=500,
        batches_per_epoch=9,
        learning_rate=1e-2,
        seed=1,
        device="cpu",
        verbose=False,
    )
    p = m.predict(ux[:100], 0 * ux[:100], uz[:100], ds.E[:100])
    np.testing.assert_allclose(p[:, 1], 0.5, atol=0.15)


@pytest.mark.filterwarnings("ignore:garf\\: an input feature is constant")
def test_weights_equal_duplication():
    """Weight 2 on an entry ~ the entry twice: same probabilities."""
    rng = np.random.default_rng(5)
    n = 20_000
    uz = rng.uniform(0.9, 1, n)
    ux = np.sqrt(1 - uz**2)
    E = rng.uniform(0.1, 0.15, n)
    window = (rng.uniform(0, 1, n) < 0.3).astype(int)  # p1 = 0.3
    w = np.where(window == 1, 3.0, 1.0)  # weighted p1 = 0.9 / (0.9 + 0.7) = 0.5625
    ds = garf.TrainingDataset(ux, 0 * ux, uz, E, window, w, metadata={"rr": 1})
    m = garf.train(
        ds,
        H=8,
        L=1,
        epochs=10,
        batch_size=500,
        learning_rate=1e-2,
        seed=1,
        device="cpu",
        verbose=False,
    )
    p = m.predict(ux[:200], 0 * ux[:200], uz[:200], E[:200])
    np.testing.assert_allclose(p[:, 1].mean(), 0.5625, atol=0.03)


@pytest.mark.filterwarnings("ignore:garf\\: an input feature is constant")
def test_russian_roulette():
    """Not-detected photons kept with probability 1/rr: predict corrects it."""
    rng = np.random.default_rng(6)
    n = 40_000
    uz = rng.uniform(0.9, 1, n)
    ux = np.sqrt(1 - uz**2)
    E = rng.uniform(0.1, 0.15, n)
    window = (rng.uniform(0, 1, n) < 0.5).astype(int)  # after RR: half detected
    ds = garf.TrainingDataset(ux, 0 * ux, uz, E, window, metadata={"rr": 10})
    m = garf.train(
        ds,
        H=8,
        L=1,
        epochs=10,
        batch_size=500,
        learning_rate=1e-2,
        seed=1,
        device="cpu",
        verbose=False,
    )
    assert m.rr == 10
    p = m.predict(ux[:200], 0 * ux[:200], uz[:200], E[:200])
    # true p1 = 0.5 / (0.5 * 10 + 0.5) = 1/11
    np.testing.assert_allclose(p[:, 1].mean(), 1 / 11, atol=0.01)


def test_rr_must_be_known_and_consistent():
    ds = dataset(n=2000)
    del ds.metadata["rr"]
    with pytest.raises(ValueError, match="rr"):
        garf.train(ds, epochs=1, device="cpu", verbose=False)
    ds.metadata["rr"] = 50
    with pytest.raises(ValueError, match="rr = 10"):
        garf.train(ds, epochs=1, rr=10, device="cpu", verbose=False)


def test_ensemble(tmp_path):
    m = garf.train(
        dataset(n=20_000),
        n_ensemble=3,
        seed=7,
        early_stopping=0,
        verbose=False,
        **dict(FAST, epochs=3),
    )
    assert m.n_ensemble == 3
    seeds = [h["seed"] for h in m.metadata["training"]["networks"]]
    assert seeds == [7, 8, 9]
    path = tmp_path / "ens.garf"
    m.save(path)
    m2 = garf.ArfModel.load(path, device="cpu")
    assert m2.n_ensemble == 3
    u = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.14])
    np.testing.assert_allclose(m2.predict(*u), m.predict(*u), rtol=1e-6)


def test_early_stopping():
    # constant learning rate, tiny problem: the validation loss stops improving
    m = garf.train(
        dataset(n=5000),
        H=8,
        L=1,
        epochs=500,
        early_stopping=5,
        lr_schedule="constant",
        learning_rate=0.05,
        batch_size=500,
        seed=3,
        device="cpu",
        verbose=False,
    )
    h = m.metadata["training"]["networks"][0]
    assert len(h["validation_loss"]) < 500
    assert len(h["validation_loss"]) - 1 - h["selected_epoch"] == 5


def test_unknown_parameter():
    with pytest.raises(ValueError, match="unknown training parameters"):
        garf.TrainingParameters.from_dict({"H": 10, "batch_per_epoch": 3})


def test_save_load_trained_model(tmp_path):
    ds = dataset(n=5000, detector={"distance_to_crystal": 74.625})
    m = garf.train(ds, seed=1, verbose=False, **dict(FAST, epochs=2))
    path = tmp_path / "m.garf"
    m.save(path)
    m2 = garf.ArfModel.load(path, device="cpu")
    assert m2.features == "ux_uy_E"
    assert m2.distance_to_crystal == 74.625
    assert m2.metadata["training"]["dataset"]["n_entries"] == 5000


def test_garf_train_cli(tmp_path):
    from click.testing import CliRunner
    from garf.bin.garf_train import garf_train

    f = tmp_path / "train.root"
    garf.write_training_dataset(f, dataset(n=5000))
    param = tmp_path / "p.json"
    param.write_text(json.dumps(dict(H=8, L=1, epochs=2, device="cpu", seed=1)))
    out = tmp_path / "m.garf"
    r = CliRunner().invoke(garf_train, [str(param), str(f), str(out), "-q"])
    assert r.exit_code == 0, r.output
    m = garf.ArfModel.load(out, device="cpu")
    assert m.rr == 1 and m.n_channels == 3
