# -*- coding: utf-8 -*-
"""Table ARF (LutArfModel, build_lut) and the folded phi features."""

import os

import numpy as np
import pytest
import torch

import garf
from test_v3_symmetric import star_dataset, star_probability


def test_folded_features():
    rng = np.random.default_rng(0)
    uz = rng.uniform(0.5, 1, 2000)
    a = rng.uniform(-np.pi, np.pi, 2000)
    s = np.sqrt(1 - uz * uz)
    E = np.full(2000, 0.14)
    params = dict(fold=6, phi0_deg=10.0)

    def f(ang):
        t = [torch.from_numpy(v) for v in (s * np.cos(ang), s * np.sin(ang), uz, E)]
        return garf.compute_features("theta_phifold_E", *t, params=params).numpy()

    x = f(a)
    assert x[:, 1].min() >= 0 and x[:, 1].max() <= 30 + 1e-9
    np.testing.assert_allclose(x[:, 0], np.degrees(np.arccos(uz)), atol=1e-10)
    # invariant under the D6 group around phi0
    for b in (a + np.pi / 3, a - 2 * np.pi / 3, 2 * np.radians(10) - a):
        np.testing.assert_allclose(f(b), x, atol=1e-9)
    # folded value: distance to the nearest axis phi0 + k 60 deg
    psi = (np.degrees(a) - 10) % 60
    np.testing.assert_allclose(x[:, 1], np.minimum(psi, 60 - psi), atol=1e-9)


def star_dataset_theta(n=600_000, seed=3, phi0=10.0):
    """As star_dataset, but theta uniform in [0, 40] deg (dense near the
    axis): the table estimate per bin does not depend on the sampling."""
    rng = np.random.default_rng(seed)
    th = np.radians(rng.uniform(0, 40, n))
    a = rng.uniform(-np.pi, np.pi, n)
    ux, uy, uz = np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.cos(th)
    p = star_probability(np.degrees(th), np.degrees(a), phi0)
    window = (rng.uniform(size=n) < p).astype(np.int64)
    E = rng.uniform(0.12, 0.16, n)
    return garf.TrainingDataset(ux, uy, uz, E, window, metadata=dict(rr=1))


def subsample_rr(ds, rr, seed=0):
    """Keep 1 / rr of the non-detected entries (as the training actor)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(ds.window)
    keep = (w > 0) | (rng.uniform(size=len(w)) < 1 / rr)
    idx = np.where(keep)[0]
    sub = ds.subset(idx)
    sub.metadata["rr"] = rr
    return sub


def lut_errors(m, phi0=10.0):
    th = np.array([0.5, 1.0, 2.5, 10.0, 15.0, 15.0, 25.0])
    ph = np.array([0.0, 30.0, 70.0, phi0, phi0 + 30, phi0 + 60, phi0])
    ux = np.sin(np.radians(th)) * np.cos(np.radians(ph))
    uy = np.sin(np.radians(th)) * np.sin(np.radians(ph))
    uz = np.cos(np.radians(th))
    p = m.predict(ux, uy, uz, np.full(len(th), 0.14))[:, 1]
    return p, star_probability(th, ph, phi0)


@pytest.mark.parametrize("rr", [1, 4])
def test_lut_recovers_probabilities(rr):
    # with rr, p = d / (d + rr n0) is limited by the kept non-detected
    # entries n0 (where p is large): more entries for rr > 1
    ds = star_dataset_theta(n=600_000 * rr, phi0=10.0)
    if rr > 1:
        ds = subsample_rr(ds, rr)
    theta_edges = np.concatenate(
        [np.arange(0, 4.01, 0.25), np.geomspace(4, 41, 12)[1:]]
    )
    m = garf.build_lut(ds, theta_edges=theta_edges, n_phi=6, n_e=1, verbose=False)
    assert m.feature_params["phi0_deg"] == pytest.approx(10.0, abs=1.5)
    p, true = lut_errors(m)
    # core: few % (interpolation); tail: statistics + binning
    np.testing.assert_allclose(p[:3], true[:3], rtol=0.1)
    np.testing.assert_allclose(p[3:], true[3:], rtol=0.3)
    assert p[3] / p[4] > 2.5  # star peak / valley (true 4)
    # statistical error estimate: of the order of the observed deviations
    err = m.relative_error()
    assert torch.isfinite(err[:16]).all() and err[:16].max() < 0.3
    # rows sum to 1
    q = m.predict(
        *[np.array([0.1, 0.2])] * 2, np.array([0.97, 0.95]), np.array([0.14, 0.14])
    )
    np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-6)


def test_lut_save_load(tmp_path):
    ds = star_dataset(n=200_000)
    m = garf.build_lut(ds, n_phi=4, n_e=2, verbose=False)
    f = tmp_path / "t.garf"
    m.save(f)
    m2 = garf.ArfModel.load(f, device="cpu")
    assert isinstance(m2, garf.LutArfModel)
    assert m2.feature_params == m.feature_params
    assert m2.metadata["training"]["method"] == "lut"
    p1, _ = lut_errors(m)
    p2, _ = lut_errors(m2)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)


def test_lut_with_projector():
    """The table works through the same interface as the networks."""
    ds = star_dataset(n=200_000)
    m = garf.build_lut(ds, n_phi=4, n_e=2, verbose=False)
    proj = garf.ArfProjector([32, 32], [4.0, 4.0], distance_to_crystal=10.0)
    rng = np.random.default_rng(1)
    n = 5000
    uz = rng.uniform(0.95, 1, n)
    a = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1 - uz * uz)
    ix, iy, values = proj.project(
        m,
        rng.uniform(-30, 30, n),
        rng.uniform(-30, 30, n),
        s * np.cos(a),
        s * np.sin(a),
        uz,
        np.full(n, 0.14),
    )
    assert values.shape[1] == m.n_windows and values.sum() > 0


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true" or not torch.backends.mps.is_available(),
    reason="needs mps",
)
def test_lut_mps_equals_cpu():
    ds = star_dataset(n=200_000)
    m = garf.build_lut(ds, n_phi=4, n_e=2, verbose=False)
    p_cpu, _ = lut_errors(m.to("cpu"))
    p_mps, _ = lut_errors(m.to("mps"))
    np.testing.assert_allclose(p_mps, p_cpu, rtol=1e-5)
