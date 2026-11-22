# -*- coding: utf-8 -*-
"""Spectral ARF: deposit classes, analytic energy windows, SpectralArfModel."""

import numpy as np
import pytest
import torch

import garf

RES = dict(method="InverseSquare", resolution=0.099, reference_kev=140.5)
WINDOWS = [(108.6, 129.6), (129.6, 150.6), (90.0, 160.0)]  # the last overlaps


def blur_mc(e_mev, rng):
    """Sampled blurring, as the GATE InverseSquare law."""
    r = 0.099 * np.sqrt(0.1405 / e_mev)
    return rng.normal(e_mev, r * e_mev / 2.3548)


def test_window_response_vs_sampling():
    rng = np.random.default_rng(0)
    for e in (0.100, 0.1295, 0.1405):
        x = blur_mc(np.full(400_000, e), rng)
        res = dict(RES, reference_mev=RES["reference_kev"] / 1000)
        W = garf.window_response(
            torch.tensor([e], dtype=torch.float64), [(a / 1000, b / 1000) for a, b in WINDOWS], res
        ).numpy()[0]
        mc = [np.mean((x >= a / 1000) & (x < b / 1000)) for a, b in WINDOWS]
        np.testing.assert_allclose(W, mc, atol=3e-3)


def synthetic_spectral(n=400_000, seed=1):
    """Detection p(theta); deposit: full peak (80 %) or a loss uniform in
    [5, 60] keV; deposit bins 1 keV from 10 to 142 keV; no RR."""
    rng = np.random.default_rng(seed)
    th = np.radians(rng.uniform(0, 6, n))
    a = rng.uniform(-np.pi, np.pi, n)
    ux, uy, uz = np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.cos(th)
    E = np.where(rng.uniform(size=n) < 0.5, 0.1405, rng.uniform(0.10, 0.141, n))
    p = 0.5 * np.exp(-0.5 * (np.degrees(th) / 2) ** 2)
    det = rng.uniform(size=n) < p
    loss = np.where(rng.uniform(size=n) < 0.8, 0.0, rng.uniform(0.005, 0.060, n))
    e = E - loss
    edges = np.arange(10, 143) / 1000.0
    k = np.digitize(e, edges)  # 1..nb inside
    window = np.where(det & (k >= 1) & (k <= len(edges) - 1), k, 0)
    md = dict(
        rr=1,
        energy_windows=[
            dict(name=f"b{i}", min_keV=edges[i] * 1000, max_keV=edges[i + 1] * 1000)
            for i in range(len(edges) - 1)
        ],
    )
    ds = garf.TrainingDataset(ux, uy, uz, E, window.astype(np.int64), metadata=md)
    return ds, (th, E, det, e)


@pytest.mark.parametrize("representation", ["loss", "absolute"])
def test_spectral_model_windows(tmp_path, representation):
    ds, (th, E, det, e) = synthetic_spectral()
    edges = garf.deposit_bins_from_metadata(ds)
    sds, info = garf.spectral_classes(ds, edges, representation)
    net = garf.train(sds, H=64, L=2, epochs=15, batch_size=4000, learning_rate=3e-3, seed=3, device="cpu", verbose=False)
    m = garf.SpectralArfModel(net, info).set_energy_windows(WINDOWS, RES)
    # expected window probabilities for photons at 140.5 keV, theta 1 deg:
    # p_det (0.8 W(E) + 0.2 mean W(E - loss))
    rng = np.random.default_rng(5)
    t = np.radians(1.0)
    pdet = 0.5 * np.exp(-0.5 * 0.25)
    loss = np.where(rng.uniform(size=300_000) < 0.8, 0.0, rng.uniform(0.005, 0.060, 300_000))
    x = blur_mc(0.1405 - loss, rng)
    expected = [pdet * np.mean((x >= a / 1000) & (x < b / 1000)) for a, b in WINDOWS]
    got = m.predict(np.array([np.sin(t)]), np.array([0.0]), np.array([np.cos(t)]), np.array([0.1405]))[0, 1:]
    np.testing.assert_allclose(got, expected, rtol=0.1)
    # save / load: the spectral description comes back, windows set again
    f = tmp_path / "s.garf"
    m.save(f)
    m2 = garf.ArfModel.load(f, device="cpu")
    assert isinstance(m2, garf.SpectralArfModel)
    m2.set_energy_windows(WINDOWS, RES)
    got2 = m2.predict(np.array([np.sin(t)]), np.array([0.0]), np.array([np.cos(t)]), np.array([0.1405]))[0, 1:]
    np.testing.assert_allclose(got2, got, rtol=1e-6)
    # the projector sees the number of windows of the configuration
    assert m2.n_windows == len(WINDOWS)
