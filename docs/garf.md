# garf: GATE ARF

garf models the **angular response function (ARF)** of a SPECT collimator-detector head with a neural network, to speed up Monte Carlo SPECT simulations in GATE.

This page documents garf from version 3. Sections marked *(v3, planned)* describe the target of the ongoing refactor and will be completed when it is implemented.

## Principle

In a full SPECT simulation, photons are tracked through the collimator and the crystal, and the digitizer turns their interactions into counts in energy windows. Most photons are absorbed in the collimator, so most of this tracking is wasted.

With ARF, the detector head is replaced by a plane in front of the collimator. A photon reaching the plane is stopped. Its energy and direction are given to a model that returns the probability of being detected in each energy window. These probabilities are added to the projection image, at the position where the photon would reach the crystal.

The model is learned once per head, collimator, radionuclide and energy windows, from a full simulation (the training dataset). Compared to the conventional histogram (table) ARF [Song 2005, Descourt 2010], the neural network depends less on the statistics of the training data and needs less training data [Sarrut 2018].

## Workflow

1. **Training dataset** (GATE, `ARFTrainingDatasetActor`): full simulation of the head; for every photon reaching the plane, store its energy and direction and the energy window where it was finally detected (or "not detected").
2. **Training** (garf, `garf_train`): train the network that predicts the window probabilities from (direction, energy).
3. **Use**:
   - in GATE, with the `ARFActor` attached to the plane: projection images,
   - or standalone, with `GarfDetector` (e.g. after a GAN source, see gaga-phsp).
4. **Validation**: compare to a short analog simulation (and/or to a table ARF, *v3, planned*).

## Conventions

### Plane frame

All quantities are expressed in the frame of the ARF plane, given by `plane_axis` in the GATE actors: the first two axes span the plane (image x and y), the third is the plane normal, pointing towards the detector. Only photons going towards the detector (`uz > 0`) are considered.

- position on the plane: `(x, y)` in mm,
- direction: unit vector `(ux, uy, uz)` (direction cosines),
- energy `E` in MeV,
- particle weight `w` (1 for analog simulations).

### Network inputs

The network does not take the direction cosines directly. The input convention says how they are converted:

| convention | inputs | used by |
| --- | --- | --- |
| `legacy` | `θ = acos(uy)`, `φ = acos(ux)` (deg), `E` | GATE 9, GATE 10 ≤ 10.0.2 training actor, models in gate_data |
| `theta_phi` | `θ = acos(uz)`, `φ = atan2(uy, ux)` (deg), `E` | GATE 10 ≥ 10.0.3 training actor |

*(v3, planned)* The convention is stored in the model file and garf computes the inputs itself; users and GATE only pass `(ux, uy, uz, E)`. The input features of new models are still to be decided (`φ = atan2` is discontinuous at ±180°).

### Energy windows and Russian roulette

The network output has one channel per class: channel 0 is "not detected", channels 1..n are the energy windows (in the order of the digitizer energy windows used for the training dataset).

Most photons are not detected, so the training dataset uses Russian roulette on them: only one out of `RR` non-detected photons is kept. At prediction, the probability of channel 0 is multiplied by `RR` and the probabilities are normalised again.

### Shift to the crystal

The model gives detection probabilities, not where the photon is detected. The count is placed at the plane position shifted towards the crystal, by `distance_to_crystal` (`d`, from the plane to the middle of the crystal):

- `legacy`: `(x, y) + d·(ux, uy)` (GATE 9 design),
- `straight` *(v3, planned)*: `(x, y) + d·(ux, uy)/uz`, the straight line to the crystal.

The two differ only at large angles (`d·sinθ` vs `d·tanθ`): negligible at low energy, where the collimator only accepts near-normal photons, but it can matter for the far halo at high energy (septal penetration).

### Image

- image size and spacing are given as `(nx, ny)`, the image is centred on the plane centre,
- pixel indices: `ix = round((x + nx·sx/2 - sx/2) / sx)`, same for `iy`; points outside the image are dropped,
- the image array is `(channels, ny, nx)`, filled as `img[c, iy, ix]`,
- with `hit_slice`, channel 0 counts the photons reaching the image (unweighted), channels 1..n the windows. Without it, the windows are channels 0..n-1.

## Training dataset (GATE)

`ARFTrainingDatasetActor`, attached to the ARF plane of a full SPECT head simulation, with a `DigitizerEnergyWindowsActor` (`energy_windows_actor`). Output (ROOT tree "ARF (training)"): `E`, `Theta`, `Phi`, `window` (0 = not detected, 1..n = energy windows: the digitizer window id + 1). Parameters: `plane_axis`, `russian_roulette`.

Recommendations:
- source: photons crossing the plane with all directions and energies that the model will see later (uniform energy up to above the highest window, isotropic towards the head),
- the source must be small or centred enough so that every photon crossing the plane can reach the crystal: photons lost at the edge of the head would teach edge losses to the network,
- energy windows must not overlap.

*(v3, planned)* The training actor also stores the direction cosines and writes the convention in the output metadata.

## Training (garf)

```
garf_train <params.json> <training_dataset.root> <output model>
```

Main parameters: `H` (neurons per layer), `L` (hidden layers), `learning_rate`, `batch_size`, `batch_per_epoch`, `epoch_max`, `early_stopping`, `RR`, `seed`, `gpu_mode`.

*(v3, planned)*: random batches over the whole dataset, learning rate decay, held-out validation set used to stop and select the model, metadata saved in the model, optional ensemble of networks trained with different seeds.

## Using a model

### In GATE: `ARFActor`

Attached to the ARF plane. Main parameters: `pth_filename`, `image_size`, `image_spacing`, `distance_to_crystal`, `batch_size`, `enable_hit_slice`, `plane_axis`, `flip_plane`, `gpu_mode`. Outputs: `counts` (one image per energy window, all runs stacked), optional `squared_counts` (for uncertainty).

### Standalone: `GarfDetector`

Projects particles (`E, x, y, z, ux, uy, uz`) on a set of detector planes (gantry rotations) and builds the projections, on CPU or GPU. Used by gaga-phsp.

### Python API *(v3, planned)*

```python
m = garf.ArfModel.load(filename)
probs = m.predict(ux, uy, uz, E)
u, v, w = m.project(x, y, ux, uy, uz, E, w=None,
                    distance_to_crystal=..., image_size=..., spacing=...,
                    shift_mode="legacy")
```

## Model file *(v3, planned)*

The current `.pth` file is a pickled dict (`torch.load(..., weights_only=False)`), needs garf's network class to be loaded, and keeps all stored epochs. The v3 format stores the selected network as named tensors (safetensors or npz) with metadata: format version, architecture (`mlp`, `n_in`, `H`, `L`, activation, `n_out`), normalisation, `RR`, input convention, energy windows, provenance (garf and GATE versions, training dataset, seed, selected epoch, validation loss). Legacy `.pth` files remain readable, and `garf_convert` converts them.

## Versions and compatibility *(v3, to be completed)*

| GATE | garf | status |
| --- | --- | --- |
| ≤ 10.0.2 | ≤ 2.11 | consistent (legacy convention) |
| 10.0.3 – 10.1.2 | ≤ 2.11 | **inconsistent**: the ARF actor passes direction cosines, garf expects angles; results are wrong, whatever the model |
| next | 3.x | to be completed |

## Tests

```
pip install -e ".[test]"
pytest
```

CPU only, a few seconds. Golden data (`tests/data/golden_v034.npz`) pin the numerical results; regenerate them with `tests/generate_golden.py` only for intended, documented changes. Known bugs are tests marked `xfail(strict=True)`: remove the marker when the bug is fixed.

## References

- Sarrut D, Krah N, Badel JN, Létang JM. Learning SPECT detector angular response function with neural network for accelerating Monte-Carlo simulations. Phys Med Biol 2018;63(20):205013. doi:10.1088/1361-6560/aae331
- Song X, Segars WP, Du Y, Tsui BMW, Frey EC. Fast modelling of the collimator-detector response in Monte Carlo simulation of SPECT imaging using the angular response function. Phys Med Biol 2005;50(8):1791-1804.
- Descourt P, Carlier T, Du Y, Song X, Buvat I, Frey EC, Bardiès M, Tsui BMW, Visvikis D. Implementation of angular response function modeling in SPECT simulations with GATE. Phys Med Biol 2010;55(9):N253-N266.
