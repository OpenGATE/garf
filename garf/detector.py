# -*- coding: utf-8 -*-
"""
Standalone ARF detector (without GATE): particles (E, position, direction, in
world coordinates) are intersected with the ARF plane of each gantry angle,
then projected with an ArfModel and an ArfProjector.

The ARF plane of angle i is at distance `radius` from the origin, with the
rotation gantry_rotations[i] * initial_plane_rotation (scipy Rotation); in
its frame, the plane is z = radius, the image axes are x and y.

Used by gaga-phsp (GAN source + ARF).
"""

import numpy as np
import torch

from .arrays import is_numpy
from .model import ArfModel, resolve_device
from .projector import ArfProjector


class GarfDetector:
    def __init__(self):
        # user parameters (set before initialize)
        self.model_filename = None
        self.model = None  # or an ArfModel (then model_filename is not used)
        self.device = "auto"
        self.batch_size = 1e5  # max number of particles per projection call
        self.image_size = None  # (nx, ny)
        self.image_spacing = None  # (sx, sy) mm
        self.distance_to_crystal = None  # mm (None: from the model)
        self.shift_mode = "legacy"
        self.radius = None  # mm, distance from the origin to the ARF plane
        self.initial_plane_rotation = None  # scipy Rotation
        self.hit_slice = False

        # computed
        self.projector = None
        self.plane_rotations = None
        self.image_arrays = None  # torch (n_angles, channels, ny, nx)
        self._matrices = None
        self._centers = None
        self._device = None
        self._dtype = None
        self.is_initialized = False

    def __str__(self):
        s = f"garf model: {self.model_filename}\n"
        s += f"garf device: {self._device or self.device}\n"
        s += f"garf batch size: {self.batch_size}\n"
        s += f"garf image size: {self.image_size} pixels, spacing {self.image_spacing} mm\n"
        s += f"garf radius: {self.radius} mm\n"
        s += f"garf distance to crystal: {self.distance_to_crystal} mm ({self.shift_mode} shift)\n"
        s += f"garf hit slice: {self.hit_slice}\n"
        if self.model is not None:
            s += f"garf features: {self.model.features}, {self.model.n_windows} energy windows, RR {self.model.rr:g}\n"
        n = len(self.plane_rotations) if self.plane_rotations is not None else 0
        s += f"garf nb of gantry angles: {n}"
        return s

    def initialize(self, gantry_rotations):
        if self.is_initialized:
            raise RuntimeError("garf: GarfDetector is already initialized")
        for name in ("image_size", "image_spacing", "radius", "initial_plane_rotation"):
            if getattr(self, name) is None:
                raise ValueError(f"garf: GarfDetector.{name} must be set")
        self._device = resolve_device(self.device)
        # mps has no float64
        self._dtype = torch.float32 if self._device.type == "mps" else torch.float64

        if self.model is None:
            if self.model_filename is None:
                raise ValueError("garf: GarfDetector needs a model or a model_filename")
            self.model = ArfModel.load(self.model_filename, device=self._device)
        else:
            self.model.to(self._device)

        self.projector = ArfProjector(
            self.image_size,
            self.image_spacing,
            distance_to_crystal=self.distance_to_crystal,
            shift_mode=self.shift_mode,
            hit_slice=self.hit_slice,
        )
        self.plane_rotations = [
            g * self.initial_plane_rotation for g in gantry_rotations
        ]
        m = np.array([r.as_matrix() for r in self.plane_rotations])
        self._matrices = torch.as_tensor(m).to(self._device, self._dtype)
        c = np.array([r.apply([0, 0, self.radius]) for r in self.plane_rotations])
        self._centers = torch.as_tensor(c).to(self._device, self._dtype)

        shape = (len(self.plane_rotations),) + self.projector.image_shape(self.model)
        self.image_arrays = torch.zeros(shape, dtype=self._dtype, device=self._device)
        self.is_initialized = True

    def plane_intersection(self, batch, i):
        """
        Particles (n, 7): E, x, y, z, ux, uy, uz (world, torch) -> dict of the
        particles reaching the ARF plane i inside the image, in the plane
        frame: x, y, ux, uy, uz, E.
        """
        e = batch[:, 0]
        pos = batch[:, 1:4]
        u = batch[:, 4:7]
        c = self._centers[i]
        n = c / torch.linalg.norm(c)
        denom = u @ n
        s = ((c - pos) @ n) / denom
        keep = (s > 0) & (denom > 0)
        q = pos[keep] + s[keep, None] * u[keep]
        m = self._matrices[i]
        local = (q - c) @ m  # = R^T (q - c)
        ld = u[keep] @ m
        ld = ld / torch.linalg.norm(ld, dim=1, keepdim=True)
        hx, hy = (w / 2 for w in self.projector.image_size_mm)
        inside = (local[:, 0].abs() < hx) & (local[:, 1].abs() < hy) & (ld[:, 2] > 0)
        return dict(
            x=local[inside, 0],
            y=local[inside, 1],
            ux=ld[inside, 0],
            uy=ld[inside, 1],
            uz=ld[inside, 2],
            E=e[keep][inside],
        )

    def add(self, batch):
        """
        Project particles (n, 7): E, x, y, z, ux, uy, uz (world coordinates,
        numpy or torch) on all the ARF planes and add the counts to the images.
        Return the number of particles that reached the images (all angles).
        """
        if not self.is_initialized:
            raise RuntimeError("garf: GarfDetector must be initialized")
        if is_numpy(batch):
            batch = torch.as_tensor(np.asarray(batch, dtype=np.float64))
        if batch.ndim != 2 or batch.shape[1] != 7:
            raise ValueError(
                f"garf: batch must be (n, 7): E, x, y, z, ux, uy, uz; shape is {tuple(batch.shape)}"
            )
        batch = batch.to(self._device, self._dtype)
        bs = int(self.batch_size)
        n_inside = 0
        with torch.no_grad():
            for start in range(0, batch.shape[0], bs):
                chunk = batch[start : start + bs]
                for i in range(len(self.plane_rotations)):
                    p = self.plane_intersection(chunk, i)
                    if p["x"].shape[0] == 0:
                        continue
                    n_inside += self.projector.accumulate(
                        self.image_arrays[i],
                        self.model,
                        p["x"],
                        p["y"],
                        p["ux"],
                        p["uy"],
                        p["uz"],
                        p["E"],
                    )
        return n_inside

    def images(self):
        """One ITK image per gantry angle: (channels, ny, nx), spacing (sx, sy, 1), centred."""
        import itk

        arrays = self.image_arrays.cpu().numpy().astype(np.float32)
        sx, sy = self.projector.image_spacing
        nx, ny = self.projector.image_size
        out = []
        for a in arrays:
            img = itk.image_from_array(np.ascontiguousarray(a))
            img.SetSpacing([sx, sy, 1.0])
            img.SetOrigin([-nx * sx / 2 + sx / 2, -ny * sy / 2 + sy / 2, 0])
            out.append(img)
        return out
