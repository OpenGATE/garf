# -*- coding: utf-8 -*-
"""
Shared fixtures for the garf tests.

All tests run on CPU, are deterministic and take seconds. The golden data
(tests/data/golden_v034.npz) pin the current numerical behaviour; they are
regenerated with tests/generate_golden.py (only when a change of the numbers
is intended and documented).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import garf

DATA = Path(__file__).parent / "data"
PTH_V034 = DATA / "arf_Tc99m_v034.pth"
GOLDEN_V034 = DATA / "golden_v034.npz"

# detector geometry used by the golden data (Intevo-like, as in test043)
DISTANCE_TO_CRYSTAL = 74.625  # mm
IMAGE_SIZE = np.array([128, 128])
IMAGE_SPACING = np.array([4.41806, 4.41806])  # mm


def legacy_angles(ux, uy):
    """GATE 9 / GATE <= 10.0.2 convention: theta = acos(uy), phi = acos(ux), deg."""
    return np.degrees(np.arccos(uy)), np.degrees(np.arccos(ux))


def make_input_batch(n=2000, seed=42):
    """
    Points on the ARF plane, in the plane frame (mm, unit direction, MeV, weight).
    Directions close to the plane normal (+z), as for a collimated detector.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-250, 250, n)
    y = rng.uniform(-250, 250, n)
    uz = rng.uniform(0.95, 1.0, n)
    a = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - uz * uz)
    ux = s * np.cos(a)
    uy = s * np.sin(a)
    # half at the Tc99m photopeak, half below (scatter)
    E = np.where(rng.uniform(0, 1, n) < 0.5, 0.1405, rng.uniform(0.05, 0.16, n))
    w = rng.uniform(0.5, 2.0, n)
    return dict(x=x, y=y, ux=ux, uy=uy, uz=uz, E=E, w=w)


def legacy_columns(b, with_weights=True):
    """Column layout expected by garf <= 2.11: x, y, theta, phi, E [, w]."""
    theta, phi = legacy_angles(b["ux"], b["uy"])
    cols = [b["x"], b["y"], theta, phi, b["E"]]
    if with_weights:
        cols.append(b["w"])
    return np.column_stack(cols)


def gate_columns(b, with_weights=True):
    """Column layout passed by the GATE ARFActor >= 10.0.3: x, y, ux, uy, uz, E [, w]."""
    cols = [b["x"], b["y"], b["ux"], b["uy"], b["uz"], b["E"]]
    if with_weights:
        cols.append(b["w"])
    return np.column_stack(cols)


@pytest.fixture(scope="session")
def golden():
    if not GOLDEN_V034.exists():
        pytest.fail(f"missing {GOLDEN_V034}, run tests/generate_golden.py")
    return dict(np.load(GOLDEN_V034))


def load_v034():
    nn, model = garf.load_nn(PTH_V034, verbose=False, gpu_mode="cpu")
    model_data = nn["model_data"]
    model_data["current_gpu_device"] = torch.device("cpu")
    model.eval()
    return nn, model, model_data


@pytest.fixture
def v034():
    """(nn, model, model_data) on CPU. Function scope: some garf functions mutate model_data."""
    return load_v034()
