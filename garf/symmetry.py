# -*- coding: utf-8 -*-
"""
Azimuthal symmetry of the detector response, measured on a training dataset.

For each theta bin, the detected fraction as a function of phi (histogram
over n_phi bins, divided by the incident entries) is decomposed in Fourier
terms; the term of order `fold` gives the amplitude (relative to the mean)
and the phase of the pattern. For hexagonal holes the response in the tails
(septal penetration) has a strong 6-fold star pattern, aligned with the hole
lattice, while the core (collimator acceptance) does not depend on phi.

phi0 (deg) is the orientation of the pattern axes. The sign of the pattern
can change with theta (Lu177 MELP: maxima at phi = 30 deg at the edge of the
acceptance, 4-6 deg, at phi = 0 further out), and cos(fold (phi - phi0))
represents both (the network multiplies it by a theta-dependent factor), so
phi0 is only defined modulo 180 / fold (30 deg for fold 6), in
[-90 / fold, 90 / fold). It is estimated from the bins where the amplitude is
significant (> 3 times its statistical noise), weighted by the detected
entries (circular mean of 2 fold phase).
"""

import numpy as np

DEFAULT_THETA_EDGES = (0, 2, 4, 6, 10, 20, 40, 90)


def measure_symmetry(ux, uy, uz, window, fold=6, theta_edges=None, n_phi=None):
    """
    ux, uy, uz: directions (plane frame), window: channel (0 = not
    detected). Returns a dict: fold, phi0_deg (None if no significant
    pattern), bins (list of dict: theta_min, theta_max, n, n_detected,
    amplitude, noise, phase_deg, significant), and the second-order
    (2-fold) amplitudes as a check of the assumption (should be noise).
    """
    theta_edges = DEFAULT_THETA_EDGES if theta_edges is None else theta_edges
    n_phi = 4 * fold if n_phi is None else n_phi
    ux, uy, uz = (np.asarray(a, dtype=np.float64) for a in (ux, uy, uz))
    detected = np.asarray(window) > 0
    theta = np.degrees(np.arccos(np.clip(uz, -1, 1)))
    phi = np.degrees(np.arctan2(uy, ux)) % 360
    centres = np.radians((np.arange(n_phi) + 0.5) * 360 / n_phi)
    bins = []
    for lo, hi in zip(theta_edges[:-1], theta_edges[1:]):
        m = (theta >= lo) & (theta < hi)
        d = m & detected
        n, nd = int(m.sum()), int(d.sum())
        b = dict(theta_min=lo, theta_max=hi, n=n, n_detected=nd)
        if nd < 10 * n_phi:
            b.update(amplitude=None, noise=None, phase_deg=None, amplitude_2=None)
            b["significant"] = False
            bins.append(b)
            continue
        h_all, _ = np.histogram(phi[m], bins=n_phi, range=(0, 360))
        h_det, _ = np.histogram(phi[d], bins=n_phi, range=(0, 360))
        r = h_det / np.maximum(h_all, 1)
        r = r / r.mean()
        c = np.mean(r * np.exp(1j * fold * centres))
        c2 = np.mean(r * np.exp(2j * centres))
        noise = np.sqrt(n_phi / nd)  # ~ amplitude of a pure noise pattern
        b.update(
            amplitude=float(2 * abs(c)),
            noise=float(noise),
            phase_deg=float(np.degrees(np.angle(c)) / fold),
            amplitude_2=float(2 * abs(c2)),
        )
        b["significant"] = bool(b["amplitude"] > 3 * noise)
        bins.append(b)
    sig = [b for b in bins if b["significant"]]
    phi0 = None
    if sig:
        # axis modulo 180 / fold: circular mean of 2 fold phase
        z = sum(
            b["n_detected"] * np.exp(2j * fold * np.radians(b["phase_deg"]))
            for b in sig
        )
        phi0 = float(np.degrees(np.angle(z)) / (2 * fold))
    return dict(fold=fold, phi0_deg=phi0, n_phi=n_phi, bins=bins)


def symmetry_table(result):
    """Text table of a measure_symmetry() result."""
    lines = [
        f"{result['fold']}-fold pattern of the detected fraction vs phi "
        f"({result['n_phi']} phi bins); amplitude relative to the mean",
        f"{'theta (deg)':>12} {'entries':>9} {'detected':>9} {'amplitude':>9} "
        f"{'noise':>6} {'phase':>7} {'2-fold':>7}",
    ]
    for b in result["bins"]:
        t = f"{b['theta_min']:g}-{b['theta_max']:g}"
        if b["amplitude"] is None:
            lines.append(f"{t:>12} {b['n']:9d} {b['n_detected']:9d}   (too few)")
            continue
        flag = " *" if b["significant"] else ""
        lines.append(
            f"{t:>12} {b['n']:9d} {b['n_detected']:9d} {b['amplitude']:9.3f} "
            f"{b['noise']:6.3f} {b['phase_deg']:7.1f} {b['amplitude_2']:7.3f}{flag}"
        )
    p = result["phi0_deg"]
    lines.append(
        f"phi0 = {p:.1f} deg, modulo {180 / result['fold']:g} (from the bins marked *)"
        if p is not None
        else "no significant pattern: phi0 not determined"
    )
    return "\n".join(lines)
