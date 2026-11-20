# -*- coding: utf-8 -*-
"""
Training on synthetic data with a known detection probability: the network
must recover it. Small network, CPU, a few seconds.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import garf
import garf.garf_train


@pytest.fixture(autouse=True)
def no_dataloader_worker(monkeypatch):
    """
    garf_train uses DataLoader(num_workers=1): on macOS (spawn) a worker process
    is started at each epoch (~6 s) and it deadlocks under pytest. Same batches,
    same order with num_workers=0.
    """

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return DataLoader(*args, **kwargs)

    monkeypatch.setattr(garf.garf_train, "DataLoader", loader)


# classes: 0 = not detected, 1 = scatter window, 2 = photopeak window
SIGMA = 8.0  # deg


def p_detected(theta, phi):
    return 0.6 * np.exp(-((theta - 90) ** 2 + (phi - 90) ** 2) / (2 * SIGMA**2))


def true_probs(theta, phi, E):
    pd = p_detected(theta, phi)
    peak = E > 0.13
    return np.column_stack((1 - pd, np.where(peak, 0, pd), np.where(peak, pd, 0)))


def synthetic_dataset(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(60, 120, n)
    phi = rng.uniform(60, 120, n)
    E = np.where(rng.uniform(0, 1, n) < 0.5, 0.1405, rng.uniform(0.05, 0.12, n))
    p = true_probs(theta, phi, E)
    r = rng.uniform(0, 1, n)
    label = (r > p[:, 0]).astype(int) + (r > p[:, 0] + p[:, 1]).astype(int)
    return np.column_stack((theta, phi, E)), label.astype(np.float64)


def train_params(**kw):
    p = dict(
        seed=1234,
        RR=1,
        H=32,
        L=1,
        learning_rate=3e-3,
        epoch_max=25,
        early_stopping=100,
        batch_size=1000,
        batch_per_epoch=1000,
        epoch_store_every=1000,
        gpu_mode="cpu",
        progress_bar=False,
    )
    p.update(kw)
    return p


def predict(nn, x):
    model = garf.Net_v1(
        nn["model_data"]["H"], nn["model_data"]["L"], nn["model_data"]["n_ene_win"]
    )
    model.load_state_dict(nn["optim"]["model_state"][-1])
    md = dict(nn["model_data"], current_gpu_device=torch.device("cpu"))
    with torch.no_grad():
        return garf.nn_predict_numpy(model, md, x)


def test_training_recovers_known_probabilities():
    torch.manual_seed(1234)
    x, y = synthetic_dataset(100_000, seed=1)
    nn = garf.train_nn(x, y, train_params())

    assert nn["model_data"]["n_ene_win"] == 3
    # evaluate on a grid away from the borders of the training range
    g = np.linspace(75, 105, 13)
    theta, phi = [a.ravel() for a in np.meshgrid(g, g)]
    errors = []
    for E in [0.1405, 0.09]:
        xe = np.column_stack((theta, phi, np.full(len(theta), E)))
        errors.append(np.abs(predict(nn, xe) - true_probs(theta, phi, xe[:, 2])))
    errors = np.concatenate(errors)
    # baseline of the garf 2.11 trainer (2026-11-20): mean 0.014, max 0.091,
    # the max is at the peak of p_detected (underfit, constant learning rate).
    # A better trainer should allow tighter bounds.
    assert errors.mean() < 0.025
    assert errors.max() < 0.12
