# -*- coding: utf-8 -*-
"""
Network input features, computed from the direction cosines and the energy.

All directions are in the frame of the ARF plane (the third axis is the plane
normal, towards the detector; only uz > 0 is meaningful).

Feature sets (columns, in this order):
- "ux_uy_E"     : ux, uy, E                               (default for new models)
- "legacy_acos" : acos(uy), acos(ux) (deg), E             (GATE 9, GATE <= 10.0.2)
- "theta_phi"   : acos(uz), atan2(uy, ux) (deg), E        (GATE 10.0.3 to 10.1.2)

Old training datasets only store two angles ("Theta", "Phi"), whose meaning
depends on the GATE version: direction_from_angles() converts them back to
direction cosines, for an explicitly given convention.
"""

import numpy as np
import torch

FEATURES = ("ux_uy_E", "legacy_acos", "theta_phi")
DEFAULT_FEATURES = "ux_uy_E"
ANGLE_CONVENTIONS = ("legacy_acos", "theta_phi")


def check_features(features):
    if features not in FEATURES:
        raise ValueError(
            f"Unknown garf features '{features}', must be one of {FEATURES}"
        )


def compute_features(features, ux, uy, uz, E):
    """
    Network inputs (n, 3) from direction cosines and energy (torch tensors,
    same dtype and device). Cosines are clamped to [-1, 1] before acos.
    """
    check_features(features)
    if features == "ux_uy_E":
        return torch.stack((ux, uy, E), dim=1)
    if features == "legacy_acos":
        theta = torch.rad2deg(torch.arccos(torch.clamp(uy, -1, 1)))
        phi = torch.rad2deg(torch.arccos(torch.clamp(ux, -1, 1)))
        return torch.stack((theta, phi, E), dim=1)
    # theta_phi
    theta = torch.rad2deg(torch.arccos(torch.clamp(uz, -1, 1)))
    phi = torch.rad2deg(torch.atan2(uy, ux))
    return torch.stack((theta, phi, E), dim=1)


def check_angles(convention, theta, phi, tolerance=1e-6):
    """
    Check that angles (numpy, deg) are consistent with the convention.
    Raise ValueError otherwise (no silent guess of the convention).
    """
    if convention not in ANGLE_CONVENTIONS:
        raise ValueError(
            f"Unknown angle convention '{convention}', must be one of {ANGLE_CONVENTIONS}"
        )
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    if len(theta) == 0:
        return
    t = tolerance
    if convention == "theta_phi":
        if theta.min() < -t or theta.max() > 90 + t:
            raise ValueError(
                f"theta_phi convention: Theta must be in [0, 90] deg "
                f"(towards the detector), found [{theta.min():.3f}, {theta.max():.3f}]. "
                f"Is this a legacy_acos dataset?"
            )
        if phi.min() < -180 - t or phi.max() > 180 + t:
            raise ValueError(
                f"theta_phi convention: Phi must be in [-180, 180] deg, "
                f"found [{phi.min():.3f}, {phi.max():.3f}]"
            )
        return
    # legacy_acos
    if min(theta.min(), phi.min()) < -t or max(theta.max(), phi.max()) > 180 + t:
        raise ValueError(
            f"legacy_acos convention: Theta and Phi must be in [0, 180] deg, "
            f"found Theta [{theta.min():.3f}, {theta.max():.3f}], "
            f"Phi [{phi.min():.3f}, {phi.max():.3f}]. Is this a theta_phi dataset?"
        )
    s = np.cos(np.radians(theta)) ** 2 + np.cos(np.radians(phi)) ** 2
    if s.max() > 1 + 1e-6:
        raise ValueError(
            f"legacy_acos convention: cos(Theta)^2 + cos(Phi)^2 must be <= 1 "
            f"(Theta = acos(uy), Phi = acos(ux)), found up to {s.max():.6f}. "
            f"Is this a theta_phi dataset?"
        )


def direction_from_angles(convention, theta, phi, check=True):
    """
    Direction cosines (numpy) from the angles (deg) of an old training dataset.
    uz >= 0 is assumed (photons going towards the detector).
    """
    if check:
        check_angles(convention, theta, phi)
    theta = np.radians(np.asarray(theta, dtype=np.float64))
    phi = np.radians(np.asarray(phi, dtype=np.float64))
    if convention == "theta_phi":
        st = np.sin(theta)
        return st * np.cos(phi), st * np.sin(phi), np.cos(theta)
    # legacy_acos
    ux = np.cos(phi)
    uy = np.cos(theta)
    uz = np.sqrt(np.clip(1 - ux * ux - uy * uy, 0, None))
    return ux, uy, uz
