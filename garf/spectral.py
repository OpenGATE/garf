# -*- coding: utf-8 -*-
"""
Spectral ARF: the network learns the energy spectrum seen by the detector
in narrow bins, not fixed energy windows. The energy windows are applied at
simulation time:

    P(window | direction, E) = sum_k p_k(direction, E) W_k(window)

with p_k the probability of a single in bin k (network output). One model
per collimator and detector, any windows (overlapping too).

Two stages where the training cuts the digitizer chain (info["stage"]):
- "measured" (universal, default): the measured energy at the end of the
  digitizer, just before the window split; W_k = fraction of bin k inside
  the window (exact when the window edges are bin edges). Works for any
  digitizer: nothing to model.
- "deposit": the deposited energy before the energy blurring; W_k =
  P(blurred e_k in window) = Phi((max - e) / s(e)) - Phi((min - e) / s(e)),
  s(e) the energy resolution (same laws as the GATE blurring actor), given
  at simulation time. Only when the measured energy is the deposit plus a
  known blur (e.g. Intevo; not the Veriton v2 charge induction).

Training data: an ARF training actor whose energy windows are contiguous
narrow bins (window id = bin; see the garf study, readme008 prototype).

Two representations of the deposit (network classes, channel 0 = no deposit
in the bins):
- "absolute": class k = bin of the deposited energy e;
- "loss": class k = bin of the energy loss E - e (the photopeak is always
  the class "loss 0", the K-escape always ~28 keV): smoother for the network.
  For the measured energy the loss can be negative (blur, gain): loss_min.
"""

import numpy as np
import torch

from .arrays import is_numpy, move, to_output, to_tensor
from .dataset import TrainingDataset
from .model import ArfModel

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
REPRESENTATIONS = ("absolute", "loss")
STAGES = ("measured", "deposit")


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


def spectral_classes(
    dataset,
    edges_mev,
    representation="loss",
    loss_bin_mev=None,
    stage="measured",
    loss_min_mev=None,
):
    """
    TrainingDataset whose window is the network class: 0 = no single in the
    bins, else 1 + k with k the energy bin ("absolute") or the energy-loss
    bin ("loss", width loss_bin_mev, default the bin width, from loss_min_mev:
    default 0 for "deposit", -0.04 MeV for "measured").
    stage: "measured" (end of the digitizer) or "deposit" (before the blur).
    Returns (dataset, info) with info the spectral description for the model.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"garf: unknown representation {representation}")
    if stage not in STAGES:
        raise ValueError(f"garf: unknown stage {stage}")
    if loss_min_mev is None:
        loss_min_mev = -0.04 if stage == "measured" else 0.0
    w = np.asarray(dataset.window)
    edges = np.asarray(edges_mev, float)
    nb = len(edges) - 1
    centre = 0.5 * (edges[1:] + edges[:-1])
    det = w > 0
    if representation == "absolute":
        cls = w.copy()
        info = dict(representation="absolute", stage=stage, edges_mev=edges.tolist())
    else:
        width = float(edges[1] - edges[0]) if loss_bin_mev is None else loss_bin_mev
        n_loss = int(np.ceil((edges[-1] - edges[0] - loss_min_mev) / width)) + 1
        E = np.asarray(dataset.E, float)
        e = np.zeros_like(E)
        e[det] = centre[w[det] - 1]
        k = np.clip(np.rint((E - e - loss_min_mev) / width).astype(int), 0, n_loss - 1)
        cls = np.where(det, 1 + k, 0)
        info = dict(
            representation="loss",
            stage=stage,
            loss_bin_mev=width,
            loss_min_mev=float(loss_min_mev),
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


def bin_window_fraction(e_mev, width_mev, windows_mev):
    """W (n, n_windows): fraction of the bin [e - width/2, e + width/2]
    (measured energy, uniform in the bin) inside each window [min, max]."""
    lo = torch.as_tensor(
        [w[0] for w in windows_mev], dtype=e_mev.dtype, device=e_mev.device
    )
    hi = torch.as_tensor(
        [w[1] for w in windows_mev], dtype=e_mev.dtype, device=e_mev.device
    )
    a = e_mev[..., None] - 0.5 * width_mev
    b = e_mev[..., None] + 0.5 * width_mev
    return ((torch.minimum(b, hi) - torch.maximum(a, lo)).clamp(min=0)) / width_mev


class SpectralArfModel(ArfModel):
    """
    Wraps a network ArfModel trained on spectral classes; set_energy_windows()
    before predict() (with the resolution for the "deposit" stage only). predict() returns (n, 1 + n_windows): channel 0 =
    1 - P(any deposit in the bins), then the window probabilities (windows
    may overlap: they do not sum to 1 then).
    """

    model_type = "spectral"

    def __init__(self, inner, info):
        super().__init__(
            inner.features,
            inner.n_channels,
            inner.rr,
            inner.metadata,
            inner.feature_params,
        )
        self.inner = inner
        self.info = dict(info)
        self.info.setdefault("stage", "deposit")  # prototype models: deposit
        self.windows_mev = None
        self.resolution = None
        if self.info["representation"] == "absolute":
            ed = np.asarray(self.info["edges_mev"])
            self._centres = torch.as_tensor(
                0.5 * (ed[1:] + ed[:-1]), dtype=torch.float64
            )
            self._width = float(ed[1] - ed[0])
        else:
            self._loss = torch.as_tensor(
                self.info.get("loss_min_mev", 0.0)
                + np.arange(self.info["n_loss"]) * self.info["loss_bin_mev"],
                dtype=torch.float64,
            )
            self._width = float(self.info["loss_bin_mev"])

    @property
    def n_windows(self):
        return 0 if self.windows_mev is None else len(self.windows_mev)

    @property
    def stage(self):
        return self.info["stage"]

    def set_energy_windows(self, windows_kev, resolution=None):
        """windows_kev: list of (min, max) in keV. resolution ("deposit"
        stage only; the measured energy is already blurred): dict(method
        'InverseSquare' | 'Linear', resolution (relative FWHM at the
        reference), reference_kev, slope_per_mev for Linear)."""
        if self.stage == "measured" and resolution is not None:
            raise ValueError(
                "garf: measured-energy model, the resolution is in the model"
            )
        if self.stage == "deposit" and resolution is None:
            raise ValueError("garf: deposited-energy model, give the energy resolution")
        self.windows_mev = [(a / 1000.0, b / 1000.0) for a, b in windows_kev]
        if resolution is not None:
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
            if self.stage == "measured":
                W = bin_window_fraction(e, self._width, self.windows_mev)
            else:
                W = window_response(e, self.windows_mev, self.resolution)  # (n, nb, nw)
            pw = torch.einsum("nb,nbw->nw", p[:, 1:], W)
            out = torch.cat([1 - p[:, 1:].sum(dim=1, keepdim=True), pw], dim=1)
        if numpy_out:
            out = out.cpu().double()
        return to_output(out, numpy_out)

    def save(self, path):
        self.inner.metadata["spectral"] = self.info
        self.inner.save(path)
