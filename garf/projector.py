# -*- coding: utf-8 -*-
"""
From photons on the ARF plane to image counts.

Conventions:
- inputs in the frame of the ARF plane: x, y (mm), ux, uy, uz (unit
  direction, uz > 0 towards the detector), E (MeV), w (weight, optional);
- image size and spacing given as (nx, ny), the image is centred on the
  plane centre;
- the count is placed at the plane position shifted towards the crystal
  by distance_to_crystal (d):
    "legacy"   : (x, y) + d * (ux, uy)          (GATE 9, garf <= 2.11)
    "straight" : (x, y) + d * (ux, uy) / uz     (straight line)
- pixel indices ix = floor((x + nx * sx / 2) / sx), same for iy; points
  outside the image are dropped;
- image array (channels, ny, nx), filled as img[c, iy, ix]; channels are the
  energy windows (model channels 1..n), preceded by the hit slice (number of
  photons reaching the image, unweighted) if hit_slice is True.
"""

import warnings

import numpy as np
import torch

from .arrays import check_same_length, is_numpy, to_output, to_tensor, move

SHIFT_MODES = ("legacy", "straight")


class ArfProjector:
    def __init__(
        self,
        image_size,
        image_spacing,
        distance_to_crystal=None,
        shift_mode="legacy",
        hit_slice=False,
    ):
        self.image_size = tuple(int(s) for s in image_size)
        self.image_spacing = tuple(float(s) for s in image_spacing)
        if len(self.image_size) != 2 or len(self.image_spacing) != 2:
            raise ValueError("garf: image_size and image_spacing must be (nx, ny)")
        if min(self.image_size) <= 0 or min(self.image_spacing) <= 0:
            raise ValueError("garf: image_size and image_spacing must be positive")
        if shift_mode not in SHIFT_MODES:
            raise ValueError(
                f"garf: unknown shift_mode '{shift_mode}', must be one of {SHIFT_MODES}"
            )
        self.distance_to_crystal = distance_to_crystal
        self.shift_mode = shift_mode
        self.hit_slice = bool(hit_slice)
        self._warned_distance = False

    @property
    def image_size_mm(self):
        return tuple(n * s for n, s in zip(self.image_size, self.image_spacing))

    def n_image_channels(self, model):
        return model.n_windows + (1 if self.hit_slice else 0)

    def image_shape(self, model):
        """(channels, ny, nx)"""
        nx, ny = self.image_size
        return self.n_image_channels(model), ny, nx

    def distance(self, model=None):
        d_user = self.distance_to_crystal
        d_model = model.distance_to_crystal if model is not None else None
        if d_user is None and d_model is None:
            raise ValueError(
                "garf: distance_to_crystal is not set, and the model does not provide it"
            )
        if d_user is None:
            return float(d_model)
        if (
            d_model is not None
            and abs(d_user - d_model) > 1e-6
            and not self._warned_distance
        ):
            warnings.warn(
                f"garf: distance_to_crystal {d_user} mm differs from the one of "
                f"the model ({d_model} mm)"
            )
            self._warned_distance = True
        return float(d_user)

    # ------------------------------------------------------------ geometry

    def _pixels(self, x, y, ux, uy, uz, d):
        if self.shift_mode == "legacy":
            px = x + d * ux
            py = y + d * uy
        else:
            px = x + d * ux / uz
            py = y + d * uy / uz
        nx, ny = self.image_size
        sx, sy = self.image_spacing
        ix = torch.floor((px + nx * sx / 2) / sx).long()
        iy = torch.floor((py + ny * sy / 2) / sy).long()
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return ix, iy, inside

    def pixels(self, x, y, ux, uy, uz, model=None):
        """Pixel indices (ix, iy) and the mask of the photons inside the image."""
        numpy_out = is_numpy(x, y, ux, uy, uz)
        x = to_tensor(x)
        y, ux, uy, uz = (to_tensor(a, like=x) for a in (y, ux, uy, uz))
        check_same_length(x=x, y=y, ux=ux, uy=uy, uz=uz)
        ix, iy, inside = self._pixels(x, y, ux, uy, uz, self.distance(model))
        return tuple(to_output(t, numpy_out) for t in (ix, iy, inside))

    # ---------------------------------------------------------- projection

    def _project(self, model, x, y, ux, uy, uz, E, w):
        ix, iy, inside = self._pixels(x, y, ux, uy, uz, self.distance(model))
        ix, iy = ix[inside], iy[inside]
        probs = model.predict(ux[inside], uy[inside], uz[inside], E[inside])
        probs = move(probs, x.device, x.dtype)
        values = probs[:, 1:]
        if w is not None:
            values = values * w[inside].unsqueeze(1)
        if self.hit_slice:
            ones = torch.ones(
                (values.shape[0], 1), dtype=values.dtype, device=values.device
            )
            values = torch.cat((ones, values), dim=1)
        return ix, iy, values

    def _inputs(self, x, y, ux, uy, uz, E, w):
        numpy_out = is_numpy(x, y, ux, uy, uz, E, w)
        x = to_tensor(x)
        y, ux, uy, uz, E, w = (to_tensor(a, like=x) for a in (y, ux, uy, uz, E, w))
        check_same_length(x=x, y=y, ux=ux, uy=uy, uz=uz, E=E, w=w)
        return numpy_out, (x, y, ux, uy, uz, E, w)

    def project(self, model, x, y, ux, uy, uz, E, w=None):
        """
        Photons inside the image: pixel indices (ix, iy) and the values to add
        per image channel (n_inside, n_image_channels): w * probabilities of
        the energy windows (and 1 for the hit slice).
        """
        numpy_out, inputs = self._inputs(x, y, ux, uy, uz, E, w)
        ix, iy, values = self._project(model, *inputs)
        return tuple(to_output(t, numpy_out) for t in (ix, iy, values))

    def accumulate(self, image, model, x, y, ux, uy, uz, E, w=None, squared_image=None):
        """
        Add the counts to image (channels, ny, nx), in place (numpy array or
        torch tensor). squared_image (optional, same shape) receives the
        squared values, for the uncertainty. Return the number of photons
        inside the image.
        """
        _, inputs = self._inputs(x, y, ux, uy, uz, E, w)
        ix, iy, values = self._project(model, *inputs)
        flat = iy * self.image_size[0] + ix
        self._add(image, flat, values, model)
        if squared_image is not None:
            self._add(squared_image, flat, values * values, model)
        return int(flat.shape[0])

    def _add(self, image, flat, values, model):
        expected = self.image_shape(model)
        if tuple(image.shape) != expected:
            raise ValueError(
                f"garf: image shape is {tuple(image.shape)}, expected "
                f"(channels, ny, nx) = {expected}"
            )
        if isinstance(image, np.ndarray):
            if not image.flags.c_contiguous or not image.flags.writeable:
                raise ValueError("garf: image must be a writeable C-contiguous array")
            img = torch.from_numpy(image)  # shares the memory
        else:
            img = image
        img = img.view(img.shape[0], -1)
        img.index_add_(
            1,
            flat.to(img.device),
            move(values, img.device, img.dtype).T,
        )
