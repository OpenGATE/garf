# -*- coding: utf-8 -*-
"""
Spectral ARF: the network learns the deposited energy in the crystal (before
the energy blurring of the digitizer), not fixed energy windows. The energy
resolution and the energy windows are applied at simulation time:

    P(window | direction, E) = sum_k p_k(direction, E) W(e_k)
    W(e) = P(blurred e in window) = Phi((max - e) / s(e)) - Phi((min - e) / s(e))

with p_k the probability of a deposit in bin k (network output) and s(e)
the energy resolution (same laws as the GATE blurring actor). One model per
collimator, any windows (overlapping too) and any resolution, without
retraining.

Training data: an ARF training actor whose energy windows are contiguous
narrow bins of the deposited energy, with no energy blurring before them
(window id = deposit bin; see the garf study, readme008 prototype).

Two representations of the deposit (network classes, channel 0 = no deposit
in the bins):
- "absolute": class k = bin of the deposited energy e;
- "loss": class k = bin of the energy loss E - e (the photopeak is always
  the class "loss 0", the K-escape always ~28 keV): smoother for the network.
"""

import numpy as np
import torch

from .arrays import is_numpy, move, to_output, to_tensor
from .dataset import TrainingDataset
from .model import ArfModel

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
REPRESENTATIONS = ("absolute", "loss")


def deposit_bins_from_metadata(dataset):
    """Bin edges (MeV) of the deposited energy from the energy windows of the
    dataset sidecar (contiguous windows, keV)."""
    ew = dataset.metadata.get("energy_windows")
    if not ew:
        raise ValueError("garf: no energy_windows in the dataset metadata")
    lo = np.array([w["min_keV"] if "min_keV" in w else w["min"] for w in ew], float)
    hi = np.array([w["max_keV"] if "max_keV" in w else w["max"] for w in ew], float)
    if np.any(np.abs(lo[1:] - hi[:-1]) > 1e-6):
        raise ValueError("garf: the deposit bins must be contiguous")
    return np.concatenate([lo, hi[-1:]]) / 1000.0


def spectral_classes(dataset, edges_mev, representation="loss", loss_bin_mev=None):
    """
    TrainingDataset whose window is the network class: 0 = no deposit in the
    bins, else 1 + k with k the deposit bin ("absolute") or the energy-loss
    bin ("loss", width loss_bin_mev, default the deposit bin width).
    Returns (dataset, info) with info the spectral description for the model.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"garf: unknown representation {representation}")
    w = np.asarray(dataset.window)
    edges = np.asarray(edges_mev, float)
    nb = len(edges) - 1
    centre = 0.5 * (edges[1:] + edges[:-1])
    det = w > 0
    if representation == "absolute":
        cls = w.copy()
        info = dict(representation="absolute", edges_mev=edges.tolist())
    else:
        width = float(edges[1] - edges[0]) if loss_bin_mev is None else loss_bin_mev
        n_loss = int(np.ceil((edges[-1] - edges[0]) / width)) + 1
        E = np.asarray(dataset.E, float)
        e = np.zeros_like(E)
        e[det] = centre[w[det] - 1]
        k = np.clip(np.rint((E - e) / width).astype(int), 0, n_loss - 1)
        cls = np.where(det, 1 + k, 0)
        info = dict(
            representation="loss",
            loss_bin_mev=width,
            n_loss=n_loss,
            deposit_min_mev=float(edges[0]),
            deposit_max_mev=float(edges[-1]),
        )
    md = dict(dataset.metadata)
    md.pop("energy_windows", None)
    md["spectral"] = info
    md["n_channels"] = 1 + (nb if representation == "absolute" else info["n_loss"])
    out = TrainingDataset(
        dataset.ux, dataset.uy, dataset.uz, dataset.E, cls, dataset.weight, md
    )
    return out, info


def window_response(e_mev, windows_mev, resolution):
    """W (n, n_windows): probability that a deposit e (MeV, tensor) is
    measured in each window [min, max] after the energy blurring."""
    method = resolution.get("method", "InverseSquare")
    r = float(resolution["resolution"])  # FWHM, relative, at the reference
    ref = float(resolution["reference_mev"])
    e = e_mev.clamp(min=1e-6)
    if method == "InverseSquare":
        fwhm_rel = r * torch.sqrt(ref / e)
    elif method == "Linear":
        fwhm_rel = float(resolution["slope_per_mev"]) * (e - ref) + r
    else:
        raise ValueError(f"garf: unknown resolution method {method}")
    s = (fwhm_rel * e * FWHM_TO_SIGMA).clamp(min=1e-9)
    lo = torch.as_tensor([w[0] for w in windows_mev], dtype=e.dtype, device=e.device)
    hi = torch.as_tensor([w[1] for w in windows_mev], dtype=e.dtype, device=e.device)
    z = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2.0)))  # noqa: E731
    return z((hi - e[..., None]) / s[..., None]) - z((lo - e[..., None]) / s[..., None])


class SpectralArfModel(ArfModel):
    """
    Wraps a network ArfModel trained on deposit classes; set_energy_windows()
    before predict(). predict() returns (n, 1 + n_windows): channel 0 =
    1 - P(any deposit in the bins), then the window probabilities (windows
    may overlap: they do not sum to 1 then).
    """

    model_type = "spectral"

    def __init__(self, inner, info):
        super().__init__(inner.features, inner.n_channels, inner.rr, inner.metadata, inner.feature_params)
        self.inner = inner
        self.info = dict(info)
        self.windows_mev = None
        self.resolution = None
        if self.info["representation"] == "absolute":
            ed = np.asarray(self.info["edges_mev"])
            self._centres = torch.as_tensor(0.5 * (ed[1:] + ed[:-1]), dtype=torch.float64)
        else:
            self._loss = torch.as_tensor(
                np.arange(self.info["n_loss"]) * self.info["loss_bin_mev"], dtype=torch.float64
            )

    @property
    def n_windows(self):
        return 0 if self.windows_mev is None else len(self.windows_mev)

    def set_energy_windows(self, windows_kev, resolution):
        """windows_kev: list of (min, max) in keV; resolution: dict(method
        'InverseSquare' | 'Linear', resolution (relative FWHM at the
        reference), reference_kev, slope_per_mev for Linear)."""
        self.windows_mev = [(a / 1000.0, b / 1000.0) for a, b in windows_kev]
        res = dict(resolution)
        res["reference_mev"] = res.pop("reference_kev") / 1000.0
        self.resolution = res
        return self

    def to(self, device):
        super().to(device)
        self.inner.to(device)
        return self

    def deposit_spectrum(self, ux, uy, uz, E):
        """Network output: (n, 1 + n_bins), probabilities of the classes."""
        return self.inner.predict(ux, uy, uz, E)

    def predict(self, ux, uy, uz, E):
        if self.windows_mev is None:
            raise ValueError("garf: set_energy_windows() first")
        numpy_out = is_numpy(ux, uy, uz, E)
        ux = to_tensor(ux)
        uy, uz, E = (to_tensor(a, like=ux) for a in (uy, uz, E))
        with torch.no_grad():
            p = self.inner.predict(ux, uy, uz, E)  # on the model device
            dev = p.device
            dt = torch.float32 if dev.type == "mps" else torch.float64
            p = p.to(dt)
            Ed = move(E, dev, dt)
            if self.info["representation"] == "absolute":
                e = move(self._centres, dev, dt)[None, :].expand(len(Ed), -1)
            else:
                e = Ed[:, None] - move(self._loss, dev, dt)[None, :]
            W = window_response(e, self.windows_mev, self.resolution)  # (n, nb, nw)
            pw = torch.einsum("nb,nbw->nw", p[:, 1:], W)
            out = torch.cat([1 - p[:, 1:].sum(dim=1, keepdim=True), pw], dim=1)
        if numpy_out:
            out = out.cpu().double()
        return to_output(out, numpy_out)

    def save(self, path):
        self.inner.metadata["spectral"] = self.info
        self.inner.save(path)
