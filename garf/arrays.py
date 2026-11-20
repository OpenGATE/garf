# -*- coding: utf-8 -*-
"""
numpy <-> torch helpers: garf computes with torch; numpy inputs give numpy
outputs, torch inputs give torch outputs (on their device).
"""

import numpy as np
import torch


def is_numpy(*arrays):
    """True if the first non-None array is not a torch tensor."""
    for a in arrays:
        if a is not None:
            return not isinstance(a, torch.Tensor)
    return True


def to_tensor(a, like=None):
    """
    numpy (or list, scalar) -> torch float64 on CPU; torch tensors are kept.
    If `like` is a tensor, the result has its device (and dtype for numpy input).
    """
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a
    t = torch.as_tensor(np.asarray(a, dtype=np.float64))
    if like is not None:
        t = t.to(device=like.device, dtype=like.dtype)
    return t


def to_output(t, numpy_out):
    if t is None:
        return None
    if numpy_out:
        return t.detach().cpu().numpy()
    return t


def check_same_length(**arrays):
    n = None
    for name, a in arrays.items():
        if a is None:
            continue
        if a.ndim != 1:
            raise ValueError(f"garf: '{name}' must be a 1D array, shape is {a.shape}")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError(
                f"garf: all inputs must have the same length, '{name}' has "
                f"{a.shape[0]} values instead of {n}"
            )
    return n
