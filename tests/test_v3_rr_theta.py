# -*- coding: utf-8 -*-
"""Direction-dependent Russian roulette and all_singles datasets (GATE
ARFTrainingDatasetActor >= 10.2: rr_theta, weight_column, edep, event_id)."""

import numpy as np
import pytest
import torch

import garf

RR_THETA = {"theta_edges_deg": [0.0, 2.0, 4.0, 90.0], "values": [2.0, 10.0, 50.0]}


def test_russian_roulette_of_theta():
    th = torch.tensor([0.0, 1.9, 2.0, 3.5, 4.0, 89.0, 95.0, -1.0], dtype=torch.float64)
    f = garf.russian_roulette_of_theta(RR_THETA, th).numpy()
    np.testing.assert_array_equal(f, [2, 2, 10, 10, 50, 50, 50, 2])


def p_det(theta_deg):
    return 0.6 * np.exp(-0.5 * (theta_deg / 2.0) ** 2)


def synthetic(n=600_000, seed=1, rr_theta=RR_THETA):
    """Detection p(theta) in window 1; the non-detected photons are kept
    with probability 1 / rr(theta), as the GATE actor."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 8, n)
    a = rng.uniform(-np.pi, np.pi, n)
    t = np.radians(th)
    ux, uy, uz = np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)
    E = rng.uniform(0.12, 0.15, n)
    det = rng.uniform(size=n) < p_det(th)
    f = garf.russian_roulette_of_theta(rr_theta, torch.from_numpy(th)).numpy()
    keep = det | (rng.uniform(size=n) < 1 / f)
    weight = np.where(det, 1.0, f)[keep]
    md = dict(rr=1.0, rr_theta=rr_theta, n_channels=2, weight_column="russian_roulette")
    return garf.TrainingDataset(
        ux[keep], uy[keep], uz[keep], E[keep], det[keep].astype(int), weight, md
    )


def test_network_rr_theta(tmp_path):
    # written and read as a GATE dataset: the factors are the loss weights
    f = tmp_path / "d.root"
    garf.write_training_dataset(f, synthetic())
    ds = garf.load_training_dataset(f)
    assert ds.weight is not None and ds.rr == 1
    m = garf.train(
        ds,
        features="theta_phi",
        H=32,
        L=2,
        epochs=8,
        batch_size=4096,
        learning_rate=3e-3,
        seed=2,
        device="cpu",
        verbose=False,
    )
    assert m.rr == 1
    th = np.array([0.5, 1.5, 2.5, 3.5, 5.0])
    t = np.radians(th)
    p = m.predict(np.sin(t), np.zeros_like(t), np.cos(t), np.full_like(t, 0.14))[:, 1]
    np.testing.assert_allclose(p, p_det(th), rtol=0.08, atol=0.004)
    f = tmp_path / "m.garf"
    m.save(f)
    m2 = garf.ArfModel.load(f, device="cpu")
    p2 = m2.predict(np.sin(t), np.zeros_like(t), np.cos(t), np.full_like(t, 0.14))[:, 1]
    np.testing.assert_allclose(p2, p, rtol=1e-6)


def test_lut_rr_theta():
    ds = synthetic()
    lut = garf.build_lut(
        ds,
        theta_edges=np.arange(0, 8.5, 0.5),
        n_phi=1,
        n_e=1,
        phi0_deg=0.0,
        verbose=False,
    )
    th = np.array([0.75, 1.75, 2.75, 4.75])
    t = np.radians(th)
    p = lut.predict(np.sin(t), np.zeros_like(t), np.cos(t), np.full_like(t, 0.14))[:, 1]
    np.testing.assert_allclose(p, p_det(th), rtol=0.06)


def test_all_singles_dataset(tmp_path):
    rng = np.random.default_rng(3)
    n = 1000
    th = np.radians(rng.uniform(0, 5, n))
    ux, uy, uz = np.sin(th), np.zeros(n), np.cos(th)
    E = np.full(n, 0.1405)
    edep = np.where(rng.uniform(size=n) < 0.5, rng.normal(0.1405, 0.006, n), 0.0)
    window = np.where((edep >= 0.1296) & (edep < 0.1506), 2, 0)
    weight = np.where(edep > 0, 1.0, 20.0)
    md = dict(rr=20.0, n_channels=3, weight_column="russian_roulette", all_singles=True)
    ds = garf.TrainingDataset(ux, uy, uz, E, window, weight, md, edep, np.arange(n))
    f = tmp_path / "d.root"
    garf.write_training_dataset(f, ds)
    d = garf.load_training_dataset(f)
    assert d.weight is None  # the Russian roulette factor, not a loss weight
    assert d.edep is not None and d.multiple_singles() == 0
    # spectral classes from edep, any bins
    edges = np.arange(100, 171, 2) / 1000.0
    sds, info = garf.spectral_classes(d, edges, "absolute")
    k = np.digitize(edep, edges)
    expected = np.where((edep > 0) & (k >= 1) & (k < len(edges)), k, 0)
    np.testing.assert_array_equal(sds.window, expected)
    assert sds.n_channels == len(edges)
    # several singles for one photon: one class per photon is not possible
    ev = np.arange(n)
    ev[11] = ev[10]
    d2 = garf.TrainingDataset(ux, uy, uz, E, window, None, md, edep, ev)
    assert d2.multiple_singles() == 1
    with pytest.raises(ValueError, match="count targets"):
        garf.train(d2, epochs=1, device="cpu", verbose=False)
    with pytest.raises(ValueError, match="count targets"):
        garf.spectral_classes(d2, edges, "absolute")
