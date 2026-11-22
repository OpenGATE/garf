# -*- coding: utf-8 -*-
"""
Table ARF: detection probabilities per energy window tabulated in
(theta, phi folded, E) bins from a training dataset, the non-network ARF of
the literature (Song et al. 2005; Descourt et al. 2010, table ARF of GATE 9).
Used as a reference for the networks: a well-populated table is an unbiased
estimate of the response, with no model assumption beyond the binning.

- Coordinates: features "theta_phifold_E": theta (deg), phi folded into the
  fundamental domain of the collimator symmetry ([0, 180 / fold] deg; all the
  symmetric directions pool their entries), E. fold = 1 with phi0 = 0 keeps
  phi in [0, 180] (mirror symmetry only).
- Estimate per bin: p_c = n_c / (sum_{c>0} n_c + rr n_0) for a window c > 0,
  p_0 = 1 - sum p_c (n: entries, or summed weights if the dataset is
  weighted): the Russian roulette is undone exactly. Empty bins: not
  detected (p_0 = 1).
- Prediction: trilinear interpolation of the probabilities between bin
  centres (constant beyond the first / last centres), torch, any device.
- Statistical error per bin (relative, of the detected probability):
  (1 - p_det) sqrt(1 / d + 1 / n0), d the detected and n0 the kept
  non-detected entries (both stored). With a Russian roulette, n0 is small
  where p is large (the core of the response): there the error comes from
  n0, not from the detected entries.
"""

import numpy as np
import torch

from .features import compute_features, resolve_feature_params
from .arrays import move
from .model import ArfModel, garf_version

DEFAULT_N_E = 24


def default_theta_edges(theta_max=90.0):
    """0.1 deg up to 4 deg (core), then logarithmic up to theta_max."""
    core = np.arange(0.0, 4.0 + 1e-9, 0.1)
    tail = np.geomspace(4.0, theta_max, 31)[1:]
    return np.concatenate([core, tail])


def default_e_edges(E, n=DEFAULT_N_E):
    """Quantiles of the energies (narrow bins where the entries are dense,
    e.g. at the emission lines), widened by a tiny margin."""
    E = np.asarray(E, dtype=np.float64)
    q = np.unique(np.quantile(E, np.linspace(0, 1, n + 1)))
    q[0], q[-1] = E.min() * (1 - 1e-9), E.max() * (1 + 1e-9)
    return q


class LutArfModel(ArfModel):
    """Table ARF; build it with garf.build_lut(dataset)."""

    model_type = "lut"

    def __init__(
        self,
        probs,
        edges,
        counts,
        counts0,
        features,
        n_channels,
        rr,
        metadata=None,
        feature_params=None,
    ):
        super().__init__(features, n_channels, rr, metadata, feature_params)
        self.probs = torch.as_tensor(probs, dtype=torch.float32)  # (nt, np, nE, nc)
        self.edges = [torch.as_tensor(e, dtype=torch.float64) for e in edges]
        self.counts = torch.as_tensor(counts, dtype=torch.float32)  # detected entries
        self.counts0 = torch.as_tensor(counts0, dtype=torch.float32)  # kept non-detected
        if self.probs.shape[-1] != self.n_channels:
            raise ValueError("garf: table and n_channels do not match")
        self._centres = [0.5 * (e[1:] + e[:-1]) for e in self.edges]

    def to(self, device):
        super().to(device)
        self.probs = self.probs.to(self.device)
        return self

    def _probs(self, x):
        # on the model device; float64 for the indices, except on mps
        dev = self.device
        dt = torch.float32 if dev.type == "mps" else torch.float64
        x = move(x, dev, dt)
        t = self.probs
        idx, frac = [], []
        for a, c in enumerate(self._centres):
            c = move(c, dev, dt)
            v = x[:, a].contiguous()
            if len(c) == 1:
                idx.append(torch.zeros(len(v), dtype=torch.long, device=dev))
                frac.append(torch.zeros_like(v))
                continue
            i = torch.clamp(torch.searchsorted(c, v) - 1, 0, len(c) - 2)
            f = torch.clamp((v - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
            idx.append(i)
            frac.append(f.to(t.dtype))
        out = 0
        for dt_, dp, de in np.ndindex(2, 2, 2):
            w = (
                (frac[0] if dt_ else 1 - frac[0])
                * (frac[1] if dp else 1 - frac[1])
                * (frac[2] if de else 1 - frac[2])
            )
            i0 = torch.clamp(idx[0] + dt_, max=t.shape[0] - 1)
            i1 = torch.clamp(idx[1] + dp, max=t.shape[1] - 1)
            i2 = torch.clamp(idx[2] + de, max=t.shape[2] - 1)
            out = out + w[:, None] * t[i0, i1, i2]
        return out / out.sum(dim=1, keepdim=True)

    def full_metadata(self):
        m = super().full_metadata()
        m.update(table_shape=list(self.probs.shape))
        return m

    def _tensors(self):
        return {
            "probs": self.probs,
            "edges_theta": self.edges[0],
            "edges_phi": self.edges[1],
            "edges_E": self.edges[2],
            "counts": self.counts,
            "counts0": self.counts0,
        }

    def relative_error(self):
        """Relative statistical error of the detected probability per bin:
        (1 - p_det) sqrt(1 / d + 1 / n0) (inf without detected entries)."""
        d, n0 = self.counts, self.counts0
        q = 1 - self.probs[..., 1:].sum(dim=-1).cpu()
        e = q * torch.sqrt(1 / d.clamp(min=1) + torch.where(n0 > 0, 1 / n0.clamp(min=1), torch.zeros_like(n0)))
        return torch.where(d > 0, e, torch.full_like(d, np.inf))


def build_lut(
    dataset,
    theta_edges=None,
    n_phi=6,
    e_edges=None,
    n_e=DEFAULT_N_E,
    fold=6,
    phi0_deg="auto",
    rr=None,
    verbose=True,
):
    """
    Table ARF from a TrainingDataset. theta_edges (deg; default: 0.1 deg up
    to 4 deg, then logarithmic to 90 deg), n_phi bins over the folded phi
    [0, 180 / fold], e_edges (default: n_e quantiles of the energies).
    fold / phi0_deg: symmetry of the collimator (phi0 'auto': measured).
    """
    from .trainer import resolve_dataset_feature_params

    log = print if verbose else (lambda *a, **k: None)
    rr = dataset.rr if dataset.rr is not None else rr
    if rr is None:
        raise ValueError("garf: the Russian roulette factor (rr) is not known")
    if dataset.rr is not None and rr != dataset.rr:
        raise ValueError(f"garf: rr = {rr} but {dataset.rr} in the dataset metadata")
    features = "theta_phifold_E"
    fp, symmetry = resolve_dataset_feature_params(
        features, dict(fold=fold, phi0_deg=phi0_deg), dataset, log
    )
    t = [torch.from_numpy(np.asarray(a, dtype=np.float64)) for a in (dataset.ux, dataset.uy, dataset.uz, dataset.E)]
    x = compute_features(features, *t, params=fp).numpy()
    if theta_edges is None:
        theta_edges = default_theta_edges(max(90.0, float(x[:, 0].max()) + 1e-6))
    phi_edges = np.linspace(0.0, 180.0 / fp["fold"], n_phi + 1)
    phi_edges[-1] += 1e-9
    if e_edges is None:
        e_edges = default_e_edges(x[:, 2], n_e)
    edges = [np.asarray(theta_edges, float), phi_edges, np.asarray(e_edges, float)]
    n_channels = dataset.n_channels
    window = np.asarray(dataset.window)
    w = np.ones(len(window)) if dataset.weight is None else np.asarray(dataset.weight, float)
    shape = tuple(len(e) - 1 for e in edges)
    ii = [np.digitize(x[:, a], edges[a]) - 1 for a in range(3)]
    inside = np.all([(i >= 0) & (i < n) for i, n in zip(ii, shape)], axis=0)
    if (~inside).sum():
        log(f"garf lut: {(~inside).sum()} entries outside the table, ignored")
    flat = np.ravel_multi_index([i[inside] for i in ii], shape)
    n = np.zeros((np.prod(shape), n_channels))
    for c in range(n_channels):
        m = window[inside] == c
        n[:, c] = np.bincount(flat[m], weights=w[inside][m], minlength=np.prod(shape))
    det = n[:, 1:].sum(axis=1)
    denom = det + rr * n[:, 0]
    probs = np.zeros_like(n)
    ok = denom > 0
    probs[ok, 1:] = n[ok, 1:] / denom[ok, None]
    probs[:, 0] = 1 - probs[:, 1:].sum(axis=1)
    probs = probs.reshape(shape + (n_channels,))
    counts = np.zeros(np.prod(shape))
    for c in range(1, n_channels):
        counts += np.bincount(flat[window[inside] == c], minlength=np.prod(shape))
    counts = counts.reshape(shape)
    counts0 = np.bincount(flat[window[inside] == 0], minlength=np.prod(shape)).reshape(shape)
    filled = (counts > 0).mean()
    log(
        f"garf lut: table {shape[0]} theta x {shape[1]} phi x {shape[2]} E x "
        f"{n_channels} channels; {filled:.0%} of the bins with detected entries, "
        f"median {np.median(counts[counts > 0]) if filled else 0:.0f} per bin"
    )
    metadata = {
        "training": {
            "method": "lut",
            "rr": rr,
            "n_entries": len(dataset),
            "dataset": dataset.metadata.get("filename"),
            "angle_convention": dataset.metadata.get("angle_convention"),
            "symmetry": symmetry,
            "fraction_of_bins_with_detected_entries": float(filled),
        },
        "provenance": {"garf_version": garf_version()},
    }
    for key in ("energy_windows", "detector"):
        if key in dataset.metadata:
            metadata[key] = dataset.metadata[key]
    return LutArfModel(
        probs,
        edges,
        counts,
        counts0,
        features,
        n_channels,
        rr,
        metadata=metadata,
        feature_params=fp,
    )
