# -*- coding: utf-8 -*-
"""Network input features and the angle conventions of old training datasets."""

import numpy as np
import pytest
import torch

import garf
from conftest import legacy_angles


def directions(n=2000, seed=0):
    """Directions towards the detector (uz > 0), all azimuths."""
    rng = np.random.default_rng(seed)
    uz = rng.uniform(0.01, 1.0, n)
    a = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1 - uz * uz)
    return s * np.cos(a), s * np.sin(a), uz


def features(name, ux, uy, uz, E):
    t = [torch.from_numpy(a) for a in (ux, uy, uz, E)]
    return garf.compute_features(name, *t).numpy()


def test_features_values():
    ux, uy, uz = directions()
    E = np.full(len(ux), 0.14)
    f = features("ux_uy_E", ux, uy, uz, E)
    np.testing.assert_array_equal(f, np.column_stack((ux, uy, E)))
    f = features("legacy_acos", ux, uy, uz, E)
    theta, phi = legacy_angles(ux, uy)
    np.testing.assert_allclose(f, np.column_stack((theta, phi, E)), atol=1e-12)
    f = features("theta_phi", ux, uy, uz, E)
    expected = np.column_stack(
        (np.degrees(np.arccos(uz)), np.degrees(np.arctan2(uy, ux)), E)
    )
    np.testing.assert_allclose(f, expected, atol=1e-12)


def test_unknown_features():
    with pytest.raises(ValueError, match="Unknown garf features"):
        garf.compute_features("theta", *[torch.zeros(1)] * 4)


@pytest.mark.parametrize("convention", ["legacy_acos", "theta_phi"])
def test_direction_from_angles_round_trip(convention):
    ux, uy, uz = directions()
    E = np.zeros(len(ux))
    f = features(convention, ux, uy, uz, E)
    vx, vy, vz = garf.direction_from_angles(convention, f[:, 0], f[:, 1])
    np.testing.assert_allclose(vx, ux, atol=1e-9)
    np.testing.assert_allclose(vy, uy, atol=1e-9)
    np.testing.assert_allclose(vz, uz, atol=1e-6)  # sqrt near uz = 0 for legacy


def test_wrong_convention_is_detected():
    """theta_phi angles read as legacy_acos (and the reverse) are rejected."""
    ux, uy, uz = directions()
    E = np.zeros(len(ux))
    tp = features("theta_phi", ux, uy, uz, E)
    with pytest.raises(ValueError, match="theta_phi dataset"):
        garf.direction_from_angles("legacy_acos", tp[:, 0], tp[:, 1])
    la = features("legacy_acos", ux, uy, uz, E)
    with pytest.raises(ValueError, match="legacy_acos dataset"):
        garf.direction_from_angles("theta_phi", la[:, 0], la[:, 1])


def test_unknown_convention():
    with pytest.raises(ValueError, match="Unknown angle convention"):
        garf.direction_from_angles("ux_uy_E", np.zeros(1), np.zeros(1))


def test_ux_uy_close_to_legacy_near_normal_incidence():
    """Near normal incidence, legacy_acos is (up to an affine map) ux, uy:
    90 - acos(u) = asin(u) = u + u^3/6 + ..., |u| < 0.1 here."""
    ux, uy, uz = directions()
    keep = uz > 0.995
    theta, phi = legacy_angles(ux[keep], uy[keep])
    np.testing.assert_allclose(90 - theta, np.degrees(uy[keep]), atol=0.02)
    np.testing.assert_allclose(90 - phi, np.degrees(ux[keep]), atol=0.02)
