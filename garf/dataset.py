# -*- coding: utf-8 -*-
"""
ARF training datasets: for every photon that reached the ARF plane in a full
simulation of the detector head, its direction (plane frame), energy, the
energy window where it was detected (0 = not detected, 1..n = windows) and
optionally its weight.

ROOT file, tree "ARF (training)", written by the GATE ARFTrainingDatasetActor:
- v3 datasets: E, ux, uy, uz, window [, weight]
- GATE >= 10.2: E, Theta, Phi, window [, weight] [, edep, event_id]: with
  all_singles, one row per single (measured energy edep before the window
  split; the rows of one photon share event_id, a photon with no single has
  edep = 0); the weight column is the Russian roulette factor of the row
  (metadata "weight_column": "russian_roulette"): constant Russian roulette
  (rr): not a loss weight, rr is corrected at prediction; direction-dependent
  (rr_theta): the factors are the loss weights, rr = 1
- old datasets: E, Theta, Phi, window (or w): the meaning of the angles
  depends on the GATE version, so the convention must be given
  (legacy_acos for GATE 9 and GATE <= 10.0.2, theta_phi for 10.0.3 to 10.1.2).

Optional metadata in a JSON file next to the ROOT file (<name>.json), e.g.
{"rr": 50, "n_channels": 3, "energy_windows": [...], "detector": {...}}.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import ANGLE_CONVENTIONS, direction_from_angles

TREE_NAME = "ARF (training)"


@dataclass
class TrainingDataset:
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    E: np.ndarray
    window: np.ndarray  # int: 0 = not detected, 1..n = energy windows
    weight: np.ndarray = None  # None = all 1
    metadata: dict = field(default_factory=dict)
    edep: np.ndarray = None  # measured energy of the single (MeV), 0 = none
    event_id: np.ndarray = None  # rows of one photon share it (contiguous)

    def __post_init__(self):
        self.window = np.asarray(self.window).astype(np.int64)
        n = len(self.E)
        for name in ("ux", "uy", "uz", "window"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"garf: dataset column '{name}' has a wrong length")
        for name in ("weight", "edep", "event_id"):
            a = getattr(self, name)
            if a is not None and len(a) != n:
                raise ValueError(f"garf: dataset column '{name}' has a wrong length")
        if n and self.window.min() < 0:
            raise ValueError("garf: negative window id in the dataset")

    def __len__(self):
        return len(self.E)

    @property
    def n_channels(self):
        """From the metadata if given, else max(window) + 1."""
        if "n_channels" in self.metadata:
            n = int(self.metadata["n_channels"])
            if len(self) and self.window.max() >= n:
                raise ValueError(
                    f"garf: window id {self.window.max()} found, but n_channels is {n}"
                )
            return n
        return int(self.window.max()) + 1

    @property
    def rr(self):
        return self.metadata.get("rr")

    def multiple_singles(self):
        """Number of rows that are not the first single of their photon
        (all_singles datasets: a photon can give several singles)."""
        if self.event_id is None or len(self) < 2:
            return 0
        ev = np.asarray(self.event_id)
        return int((ev[1:] == ev[:-1]).sum())

    def subset(self, index):
        def sub(a):
            return None if a is None else a[index]

        return TrainingDataset(
            self.ux[index],
            self.uy[index],
            self.uz[index],
            self.E[index],
            self.window[index],
            sub(self.weight),
            dict(self.metadata),
            sub(self.edep),
            sub(self.event_id),
        )

    def split(self, validation_fraction, seed=None):
        """Random (train, validation) split."""
        if not 0 <= validation_fraction < 1:
            raise ValueError("garf: validation_fraction must be in [0, 1)")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        n_val = int(round(validation_fraction * len(self)))
        return self.subset(np.sort(perm[n_val:])), self.subset(np.sort(perm[:n_val]))

    def summary(self):
        counts = np.bincount(self.window, minlength=self.n_channels)
        s = f"{len(self)} entries, {self.n_channels} channels, counts per channel {counts.tolist()}"
        s += f", E in [{self.E.min() * 1000:.1f}, {self.E.max() * 1000:.1f}] keV"
        if self.rr is not None:
            s += f", RR = {self.rr}"
        return s


def load_training_dataset(filename, angle_convention=None):
    """
    Read a training dataset ROOT file (v3 or old format, see the module doc).
    For datasets with Theta, Phi, the angle convention is angle_convention if
    given, else the one of the sidecar JSON file (key "angle_convention"),
    else an error. Ignored for datasets with ux, uy, uz.
    """
    import uproot

    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"garf: training dataset '{filename}' not found")
    with uproot.open(filename) as f:
        name = TREE_NAME
        if TREE_NAME not in f:
            # the tree is named after the training actor: a single ARF tree is used
            trees = sorted({k.split(";")[0] for k in f.keys(cycle=False)})
            trees = [t for t in trees if t.startswith("ARF")]
            if len(trees) != 1:
                raise ValueError(
                    f"garf: '{filename}' has no tree '{TREE_NAME}', keys are {f.keys()}"
                )
            name = trees[0]
        tree = f[name]
        keys = set(tree.keys())
        a = tree.arrays(library="np")

    window_key = "window" if "window" in keys else "w" if "w" in keys else None
    if window_key is None or "E" not in keys:
        raise ValueError(f"garf: '{filename}' needs E and window columns, has {keys}")
    window = np.rint(a[window_key]).astype(np.int64)

    metadata = {"filename": filename.name}
    sidecar = filename.with_suffix(".json")
    if sidecar.is_file():
        metadata.update(json.loads(sidecar.read_text()))

    if {"ux", "uy", "uz"} <= keys:
        ux, uy, uz = a["ux"], a["uy"], a["uz"]
        metadata["format"] = "directions"
    elif {"Theta", "Phi"} <= keys:
        if angle_convention is None:
            # written by the GATE training actor (>= 10.2) in the sidecar
            angle_convention = metadata.get("angle_convention")
        if angle_convention is None:
            raise ValueError(
                f"garf: '{filename}' is an old dataset (Theta, Phi): the angle "
                f"convention must be given, one of {ANGLE_CONVENTIONS} "
                f"(legacy_acos: GATE 9 and GATE <= 10.0.2; theta_phi: GATE 10.0.3 to 10.1.2)"
            )
        ux, uy, uz = direction_from_angles(angle_convention, a["Theta"], a["Phi"])
        metadata["format"] = "angles"
        metadata["angle_convention"] = angle_convention
    else:
        raise ValueError(
            f"garf: '{filename}' needs ux, uy, uz (or Theta, Phi) columns, has {keys}"
        )
    weight = a["weight"] if "weight" in keys else None
    if metadata.get("weight_column") == "russian_roulette":
        if metadata.get("rr_theta") is None:
            # constant Russian roulette: corrected with rr at prediction
            weight = None
        else:
            # direction-dependent Russian roulette: the factors are loss
            # weights (importance weights), the network learns the
            # detection probabilities directly (a correction after the
            # training would need the network to follow the steps of rr)
            metadata["rr"] = 1.0
    edep = a["edep"] if "edep" in keys else None
    event_id = a["event_id"] if "event_id" in keys else None
    return TrainingDataset(ux, uy, uz, a["E"], window, weight, metadata, edep, event_id)


def write_training_dataset(filename, dataset):
    """Write a v3 training dataset (ROOT) and its metadata (JSON), e.g. for tests."""
    import uproot

    filename = Path(filename)
    cols = {
        "E": dataset.E,
        "ux": dataset.ux,
        "uy": dataset.uy,
        "uz": dataset.uz,
        "window": dataset.window.astype(np.float64),
    }
    for name in ("weight", "edep", "event_id"):
        if getattr(dataset, name) is not None:
            cols[name] = getattr(dataset, name)
    with uproot.recreate(filename) as f:
        f[TREE_NAME] = cols
    meta = {
        k: v for k, v in dataset.metadata.items() if k not in ("filename", "format")
    }
    if meta:
        filename.with_suffix(".json").write_text(json.dumps(meta, indent=2))
