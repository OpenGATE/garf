# -*- coding: utf-8 -*-
"""
Training of the ARF network from a TrainingDataset.

- every epoch is one pass over the training data, in a new random order
  (or `batches_per_epoch` random batches);
- cross-entropy loss, weighted by the particle weights if the dataset has them;
- Adam, learning rate with cosine decay (or constant);
- a held-out validation set: its loss selects the network (best epoch) and
  stops the training (`early_stopping` epochs without improvement);
- optional ensemble: N networks with different seeds, same data and
  normalisation, whose probabilities are averaged at prediction.
"""

import copy
import warnings
import socket
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .features import DEFAULT_FEATURES, check_features, compute_features
from .model import BLOCKS, MlpArfModel, garf_version, resolve_device


@dataclass
class TrainingParameters:
    features: str = DEFAULT_FEATURES
    H: int = 200
    L: int = 3
    block: str = "mlp"  # "mlp" | "residual" (L blocks of 2 layers)
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    epochs: int = 100
    batch_size: int = 2000
    batches_per_epoch: int = None  # None: one pass over the data
    validation_fraction: float = 0.1
    early_stopping: int = 20  # epochs without improvement; 0: never stop
    seed: int = None  # None: random
    n_ensemble: int = 1
    device: str = "auto"
    rr: float = None  # Russian roulette of the dataset (if not in its metadata)

    def __post_init__(self):
        check_features(self.features)
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"garf: unknown lr_schedule '{self.lr_schedule}'")
        if self.block not in BLOCKS:
            raise ValueError(f"garf: unknown block '{self.block}', must be one of {BLOCKS}")
        if self.n_ensemble < 1:
            raise ValueError("garf: n_ensemble must be >= 1")

    @classmethod
    def from_dict(cls, d):
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"garf: unknown training parameters {sorted(unknown)}, "
                f"known: {sorted(known)}"
            )
        return cls(**d)


def _inputs(dataset, features):
    t = [
        torch.from_numpy(np.asarray(a, dtype=np.float64))
        for a in (dataset.ux, dataset.uy, dataset.uz, dataset.E)
    ]
    return compute_features(features, *t)


def _loss(net, x, y, w):
    ce = F.cross_entropy(net(x), y, reduction="none")
    if w is None:
        return ce.mean()
    return (ce * w).sum() / w.sum()


def _eval_loss(net, x, y, w, batch_size):
    if x.shape[0] == 0:
        return float("nan")
    net.eval()
    total, norm = 0.0, 0.0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            sl = slice(s, s + batch_size)
            ce = F.cross_entropy(net(x[sl]), y[sl], reduction="none")
            if w is None:
                total += ce.sum().item()
                norm += ce.shape[0]
            else:
                total += (ce * w[sl]).sum().item()
                norm += w[sl].sum().item()
    net.train()
    return total / norm


def train(dataset, params=None, verbose=True, **kwargs):
    """
    Train an MlpArfModel from a TrainingDataset. Parameters: a
    TrainingParameters, or keyword arguments (see TrainingParameters).
    """
    if params is None:
        params = TrainingParameters(**kwargs)
    elif kwargs:
        raise ValueError("garf: give either params or keyword arguments, not both")
    p = params
    log = print if verbose else (lambda *a, **k: None)

    rr = dataset.rr
    if rr is None:
        rr = p.rr
    elif p.rr is not None and p.rr != rr:
        raise ValueError(
            f"garf: rr = {p.rr} in the parameters but {rr} in the dataset metadata"
        )
    if rr is None:
        raise ValueError("garf: the Russian roulette factor (rr) is not known")

    seed = (
        p.seed if p.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    )
    device = resolve_device(p.device)
    n_channels = dataset.n_channels

    train_set, val_set = dataset.split(p.validation_fraction, seed=seed)
    x_train = _inputs(train_set, p.features)
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0)
    if torch.any(x_std == 0):
        warnings.warn("garf: an input feature is constant in the training data")
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)

    def prepare(ds, x):
        x = ((x - x_mean) / x_std).float().to(device)
        y = torch.from_numpy(ds.window).to(device)
        w = (
            None
            if ds.weight is None
            else torch.from_numpy(np.asarray(ds.weight, dtype=np.float32)).to(device)
        )
        return x, y, w

    xt, yt, wt = prepare(train_set, x_train)
    xv, yv, wv = prepare(val_set, _inputs(val_set, p.features))
    use_val = len(val_set) > 0
    log(
        f"garf train: {len(train_set)} training, {len(val_set)} validation entries, "
        f"{n_channels} channels, features {p.features}, device {device}, seed {seed}"
    )

    architecture = dict(
        n_in=3, H=p.H, L=p.L, n_out=n_channels, activation="relu", block=p.block
    )
    nets, histories = [], []
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    for k in range(p.n_ensemble):
        net, history = _train_one(
            architecture, xt, yt, wt, xv, yv, wv, use_val, p, seed + k, device, log
        )
        nets.append(net)
        histories.append(history)

    training = {
        "params": asdict(p),
        "seed": seed,
        "dataset": {
            "filename": dataset.metadata.get("filename"),
            "n_entries": len(dataset),
            "n_train": len(train_set),
            "n_validation": len(val_set),
            "weighted": dataset.weight is not None,
            "angle_convention": dataset.metadata.get("angle_convention"),
        },
        "networks": histories,
        "start": start,
        "end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
    }
    metadata = {"training": training, "provenance": {"garf_version": garf_version()}}
    for key in ("energy_windows", "detector"):
        if key in dataset.metadata:
            metadata[key] = dataset.metadata[key]
    model = MlpArfModel(
        [n.cpu() for n in nets],
        x_mean,
        x_std,
        features=p.features,
        n_channels=n_channels,
        rr=rr,
        architecture=architecture,
        metadata=metadata,
    )
    return model.to("cpu")


def _train_one(architecture, xt, yt, wt, xv, yv, wv, use_val, p, seed, device, log):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = architecture
    net = MlpArfModel.build_net(a).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=p.learning_rate)
    scheduler = None
    if p.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=p.epochs
        )

    n = xt.shape[0]
    n_batches = max(1, n // p.batch_size)
    if p.batches_per_epoch is not None:
        n_batches = p.batches_per_epoch
    best = dict(loss=float("inf"), epoch=-1, state=None)
    history = dict(seed=seed, train_loss=[], validation_loss=[], learning_rate=[])
    net.train()
    for epoch in range(p.epochs):
        if p.batches_per_epoch is None:
            perm = torch.randperm(n, generator=gen)
        else:
            perm = torch.randint(0, n, (n_batches * p.batch_size,), generator=gen)
        total, count = 0.0, 0
        for b in range(n_batches):
            idx = perm[b * p.batch_size : (b + 1) * p.batch_size].to(device)
            if idx.shape[0] == 0:
                break
            loss = _loss(net, xt[idx], yt[idx], None if wt is None else wt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * idx.shape[0]
            count += idx.shape[0]
        train_loss = total / count
        val_loss = _eval_loss(net, xv, yv, wv, 65536) if use_val else train_loss
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        if val_loss < best["loss"]:
            best = dict(
                loss=val_loss, epoch=epoch, state=copy.deepcopy(net.state_dict())
            )
            log(
                f"  epoch {epoch}: train {train_loss:.6f}, validation {val_loss:.6f} (best)"
            )
        elif p.early_stopping and epoch - best["epoch"] >= p.early_stopping:
            log(f"  epoch {epoch}: {p.early_stopping} epochs without improvement, stop")
            break
    net.load_state_dict(best["state"])
    history["selected_epoch"] = best["epoch"]
    history["selected_loss"] = best["loss"]
    history["selection"] = "validation" if use_val else "train"
    log(f"  selected epoch {best['epoch']}, loss {best['loss']:.6f}")
    return net, history
