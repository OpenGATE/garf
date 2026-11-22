# -*- coding: utf-8 -*-
"""
"symmetric" features (log theta, cos(fold (phi - phi0)) tanh(theta / theta_core), E)
and the measurement of the azimuthal symmetry of a training dataset.
"""

import numpy as np
import pytest
import torch

import garf

FAST = dict(H=32, L=2, epochs=40, learning_rate=1e-2, batch_size=2000, device="cpu")


def star_probability(theta_deg, phi_deg, phi0=10.0, fold=6):
    """Detection probability: narrow core + tail with a fold-fold star."""
    core = 0.6 * np.exp(-0.5 * (theta_deg / 1.5) ** 2)
    tail = 0.05 * np.exp(-theta_deg / 15) * (1 + 0.6 * np.cos(np.radians(fold * (phi_deg - phi0))))
    return core + tail * (theta_deg > 2)


def star_dataset(n=300_000, seed=3, phi0=10.0, fold=6):
    rng = np.random.default_rng(seed)
    # uniform in solid angle, theta up to 40 deg
    uz = rng.uniform(np.cos(np.radians(40)), 1.0, n)
    a = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1 - uz * uz)
    ux, uy = s * np.cos(a), s * np.sin(a)
    theta = np.degrees(np.arccos(uz))
    p = star_probability(theta, np.degrees(a), phi0, fold)
    E = rng.uniform(0.12, 0.16, n)
    window = (rng.uniform(size=n) < p).astype(np.int64)  # 1 window
    return garf.TrainingDataset(ux, uy, uz, E, window, metadata=dict(rr=1))


def test_symmetric_feature_values():
    rng = np.random.default_rng(0)
    uz = rng.uniform(0.5, 1, 1000)
    a = rng.uniform(-np.pi, np.pi, 1000)
    s = np.sqrt(1 - uz * uz)
    ux, uy, E = s * np.cos(a), s * np.sin(a), np.full(1000, 0.14)
    params = dict(fold=6, phi0_deg=10.0, theta0_deg=0.2, theta_core_deg=3.0)
    t = [torch.from_numpy(v) for v in (ux, uy, uz, E)]
    f = garf.compute_features("symmetric", *t, params=params).numpy()
    theta = np.degrees(np.arccos(uz))
    expected = np.column_stack(
        (
            np.log(theta + 0.2),
            np.cos(6 * (a - np.radians(10))) * np.tanh(theta / 3),
            E,
        )
    )
    np.testing.assert_allclose(f, expected, atol=1e-10)
    # invariance: rotation by 60 deg and mirror about phi0
    for b in (a + np.pi / 3, 2 * np.radians(10) - a):
        t2 = [torch.from_numpy(v) for v in (s * np.cos(b), s * np.sin(b), uz, E)]
        f2 = garf.compute_features("symmetric", *t2, params=params).numpy()
        np.testing.assert_allclose(f2, f, atol=1e-10)
    # on the axis: finite, no phi dependence
    z = torch.tensor([0.0], dtype=torch.float64)
    f0 = garf.compute_features("symmetric", z, z, z + 1, z + 0.14).numpy()
    assert np.all(np.isfinite(f0)) and f0[0, 1] == 0


def test_feature_params_resolution():
    p = garf.resolve_feature_params("symmetric", dict(fold=4))
    assert p == dict(fold=4, phi0_deg=0.0, theta0_deg=0.2, theta_core_deg=3.0)
    assert garf.resolve_feature_params("ux_uy_E") is None
    with pytest.raises(ValueError, match="unknown parameters"):
        garf.resolve_feature_params("symmetric", dict(folds=6))
    with pytest.raises(ValueError, match="take no parameters"):
        garf.resolve_feature_params("ux_uy_E", dict(fold=6))
    with pytest.raises(ValueError, match="unknown parameters"):
        garf.TrainingParameters(features="symmetric", feature_params=dict(x=1))


@pytest.mark.parametrize("phi0", [10.0, -25.0])
def test_measure_symmetry(phi0):
    ds = star_dataset(phi0=phi0)
    r = garf.measure_symmetry(ds.ux, ds.uy, ds.uz, ds.window, fold=6)
    # phi0 modulo 30 deg, in [-15, 15)
    expected = (phi0 + 15) % 30 - 15
    assert r["phi0_deg"] == pytest.approx(expected, abs=1.5)
    core = r["bins"][0]  # 0-2 deg: no pattern
    assert not core["significant"]
    tail = [b for b in r["bins"] if b["theta_min"] >= 10 and b["amplitude"] is not None]
    assert tail and all(b["significant"] for b in tail)
    assert "phi0" in garf.symmetry_table(r)


def test_measure_symmetry_no_pattern():
    ds = star_dataset()
    rng = np.random.default_rng(5)
    a = rng.uniform(-np.pi, np.pi, len(ds))  # random phi: pattern destroyed
    s = np.sqrt(1 - ds.uz**2)
    r = garf.measure_symmetry(s * np.cos(a), s * np.sin(a), ds.uz, ds.window)
    assert r["phi0_deg"] is None


def test_train_symmetric_auto_phi0(tmp_path):
    ds = star_dataset(n=1_000_000, phi0=10.0)
    m = garf.train(
        ds, features="symmetric", seed=7, early_stopping=0, verbose=False, **FAST
    )
    assert m.features == "symmetric"
    assert m.feature_params["phi0_deg"] == pytest.approx(10.0, abs=1.5)
    assert m.metadata["training"]["symmetry"]["phi0_deg"] is not None
    # the model reproduces the star in the tail (peak / valley ratio 1.6 / 0.4)
    th = np.full(2, 15.0)
    ph = np.array([10.0, 40.0])
    ux = np.sin(np.radians(th)) * np.cos(np.radians(ph))
    uy = np.sin(np.radians(th)) * np.sin(np.radians(ph))
    uz = np.cos(np.radians(th))
    p = m.predict(ux, uy, uz, np.full(2, 0.14))[:, 1]
    true = star_probability(th, ph)
    np.testing.assert_allclose(p, true, rtol=0.3)
    assert p[0] / p[1] > 2.5  # true ratio 4
    # save / load keeps the feature parameters
    f = tmp_path / "s.garf"
    m.save(f)
    m2 = garf.ArfModel.load(f, device="cpu")
    assert m2.feature_params == m.feature_params
    np.testing.assert_allclose(m2.predict(ux, uy, uz, np.full(2, 0.14)), m.predict(ux, uy, uz, np.full(2, 0.14)))


def test_train_symmetric_fixed_phi0():
    ds = star_dataset(n=50_000, phi0=10.0)
    m = garf.train(
        ds,
        features="symmetric",
        feature_params=dict(phi0_deg=0.0),
        seed=7,
        verbose=False,
        **dict(FAST, epochs=2),
    )
    assert m.feature_params["phi0_deg"] == 0.0
    assert m.metadata["training"]["symmetry"] is None
