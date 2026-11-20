# garf tests

The tests check the numerical behaviour of garf: geometry, network output, images and training. They were written before the v3 refactor to pin the behaviour of garf 2.11, so that any later change of the numbers is either detected or intended.

- 35 tests: 30 pass, 5 are known bugs (`xfail`), see below.
- CPU only, deterministic, about 6 s in total.
- No GATE needed. Tests that need a GATE simulation (analog vs ARF) belong to the opengate test suite.

## Running

```
pip install -e ".[test]"       # adds pytest and scipy
pytest                          # all tests (testpaths = tests, in pyproject.toml)
pytest -rxX                     # also list the known bugs
pytest tests/test_image.py -k weights
pytest --runxfail tests/test_known_bugs.py   # see how the known bugs fail
```

CI: `.github/workflows/tests.yml` runs `pytest -rxX` on ubuntu and macOS, Python 3.10 and 3.13, with the CPU build of torch.

## Files

| file | content | tests |
| --- | --- | --- |
| `tests/conftest.py` | shared helpers, fixtures and constants | — |
| `tests/test_golden.py` | reference results of a real model | 3 |
| `tests/test_geometry.py` | angles, shift to the crystal, plane intersection | 12 |
| `tests/test_image.py` | network output normalisation, pixels, image accumulation | 14 |
| `tests/test_train.py` | training on synthetic data | 1 |
| `tests/test_known_bugs.py` | known bugs, `xfail(strict=True)` | 5 |
| `tests/generate_golden.py` | script that writes the golden data (not a test) | — |
| `tests/data/arf_Tc99m_v034.pth` | legacy model (Tc99m, 3 channels, H=200, L=3), from gate_data | — |
| `tests/data/golden_v034.npz` | inputs and reference outputs for `test_golden.py` | — |

## Shared helpers (`conftest.py`)

- Detector used by the golden data: `DISTANCE_TO_CRYSTAL = 74.625 mm`, `IMAGE_SIZE = 128 × 128`, `IMAGE_SPACING = 4.41806 mm` (Intevo-like, as in the GATE test043).
- `make_input_batch(n=2000, seed=42)`: points on the plane in the plane frame. `x, y` uniform in ±250 mm; directions close to the normal (`uz` in [0.95, 1], uniform azimuth); `E` half at the Tc99m photopeak (0.1405 MeV), half uniform in [0.05, 0.16] MeV; weights uniform in [0.5, 2].
- `legacy_angles(ux, uy)`: `θ = acos(uy)`, `φ = acos(ux)` in degrees (legacy convention).
- `legacy_columns(batch)`: the layout read by garf ≤ 2.11, `x, y, θ, φ, E [, w]`.
- `gate_columns(batch)`: the layout passed by the GATE `ARFActor` ≥ 10.0.3, `x, y, ux, uy, uz, E [, w]`.
- Fixtures: `golden` (the content of `golden_v034.npz`), `v034` (the model loaded on CPU, in eval mode; function scope because some garf functions modify `model_data`).

## Golden tests (`test_golden.py`)

The batch of `make_input_batch()` (stored in the npz, so it does not depend on the numpy random generator) goes through the real model `arf_Tc99m_v034.pth` with the legacy layout and weights.

| test | checks |
| --- | --- |
| `test_golden_predict` | `nn_predict_numpy`: probabilities per channel (2000 × 3) equal the reference |
| `test_golden_points_to_image` | `arf_from_points_to_image_counts`: pixel indices `u, v` identical, weighted probabilities equal; then `image_from_coordinates_add_numpy` (with hit slice) gives the reference image |
| `test_golden_probabilities_are_normalised` | the reference probabilities sum to 1 and are ≥ 0 |

Tolerance: `rtol = 1e-5`, `atol = 1e-8` (the network runs in float32, results can differ slightly between platforms or torch versions); pixel indices must be identical.

## Geometry (`test_geometry.py`)

| test | checks |
| --- | --- |
| `test_angle_offset_legacy_is_d_times_direction` | with legacy angles, `compute_angle_offset_numpy` shifts by exactly `d·(ux, uy)` |
| `test_angle_offset_normal_incidence_is_zero` | θ = φ = 90° gives no shift |
| `test_angle_offset_torch_equals_numpy` | torch and numpy versions give the same shift |
| `test_plane_intersection_numpy[0, 30, 90, 180]` | `arf_plane_intersection` (numpy, used by `GarfDetector`) for a head rotated around y: position on the plane, legacy angles and energy equal an independent straight-line computation; photons outside the image are dropped |
| `test_plane_intersection_torch_equals_numpy[0, 30, 90, 180]` | same with `GarfDetectorPlane.plane_intersection` (torch, float32: tolerance 1e-3 mm, 1e-2 deg) |
| `test_input_batch_directions_are_unit` | sanity check of the test batch |

The reference for the intersection is computed in the plane frame: the source points (within ±50 mm of the origin, directions with `uz ≥ 0.8`) are rotated, the line is intersected with the plane at distance 300 mm, then the points outside the 128 × 4.41806 mm image are removed.

## Image (`test_image.py`)

Most tests use a constant network (`ConstantModel`: returns `log(p)` whatever the input, so the softmax gives back `p`). The expected values are then exact and do not depend on a trained model.

| test | checks |
| --- | --- |
| `test_predict_russian_roulette_normalisation[RR = 1, 50]` | output = `p` with channel 0 multiplied by `RR`, normalised again |
| `test_predict_accepts_lower_case_rr_key` | `model_data["rr"]` is accepted as well as `"RR"` |
| `test_pixel_centres_map_to_their_pixel[128×128, 64×32]` | a point at the centre of pixel `(ix, iy)`, normal incidence, gives `v = ix`, `u = iy`, also for a non-square image with different spacings |
| `test_points_outside_are_removed[128×128, 64×32]` | points just outside each border are dropped, a point just inside a corner is kept in the corner pixel |
| `test_particle_weights_multiply_probabilities` | with a weight column (6 columns), probabilities are multiplied by the weights |
| `test_image_accumulation_numpy[16×16, 16×8] × [hit slice or not]` | `image_from_coordinates_add_numpy` against an explicit loop: image `(channels, ny, nx)`, `img[c, iy, ix]`; with hit slice, channel 0 counts the hits (unweighted) |
| `test_torch_image_equals_numpy_square[hit slice or not]` | the torch image path of `GarfDetector` (`build_image_from_projected_points_torch`) gives the same image as the numpy path, 32 × 32 image, angles in [80°, 100°] (so the shift to the crystal is tested too) |

## Training (`test_train.py`)

`test_training_recovers_known_probabilities`: a synthetic training dataset with a known answer.

- Inputs: θ, φ uniform in [60°, 120°], `E` half at 0.1405 MeV, half in [0.05, 0.12] MeV; 100 000 entries.
- Truth: detection probability `0.6·exp(-((θ-90)² + (φ-90)²) / (2·8²))`; when detected, channel 2 (photopeak) if `E > 0.13`, else channel 1 (scatter); otherwise channel 0.
- Network H = 32, L = 1, 25 epochs, learning rate 3e-3, batches of 1000, `RR = 1`.
- Check on a 13 × 13 grid of θ, φ in [75°, 105°], for both energies: mean absolute error on the probabilities < 0.025, max < 0.12.

With the garf 2.11 trainer: mean 0.014, max 0.091 (at the peak of the detection probability, which the network underfits). These bounds are a baseline; an improved trainer should allow tighter ones.

The file has an autouse fixture that replaces the `DataLoader` of `garf_train` by one with `num_workers=0`: with `num_workers=1`, a worker process starts at every epoch (about 6 s on macOS) and it deadlocks under pytest. The batches and their order are unchanged.

## Known bugs (`test_known_bugs.py`)

Each test describes the **correct** behaviour and is marked `xfail(strict=True)`. Today they fail. When a bug is fixed, the test passes, strict mode reports it as an error (`XPASS(strict)`), and the marker must be removed: the test then stays as a normal regression test.

| test | bug | how it fails today |
| --- | --- | --- |
| `test_gate_column_layout_gives_golden_result` | GATE ≥ 10.0.3 passes `x, y, ux, uy, uz, E, w`, garf reads `x, y, θ, φ, E` (and the weights only with 6 columns) | 337 of 2000 points leave the image (shift of about `d` on both axes), and the others have wrong probabilities |
| `test_torch_image_equals_numpy_non_square[hit slice or not]` | torch image path: `x` checked against `size[1]`, flat index `y·shape[0] + x` | `IndexError` with a 32 × 16 image |
| `test_load_nn_returns_best_epoch` | `load_nn` looks for `best_epoch_eval`, never written: always the last stored state | a file with two states and `best_epoch_index = 0` loads the second one |
| `test_training_uses_the_whole_dataset` | no shuffle and the epoch stops after `batch_per_epoch` (+1) batches: only the first entries are used | ordered dataset (first half channel 0, second half channel 1): predicted probability of channel 1 is 0.003 instead of 0.5 |

## Golden data

`tests/generate_golden.py` writes `tests/data/golden_v034.npz`: the inputs (`in_x, in_y, in_ux, in_uy, in_uz, in_E, in_w`) and the outputs of the current code (`probs`, `u`, `v`, `w_pred`, `image`).

Regenerate only when a change of the numbers is intended, and say why in the commit:

```
python tests/generate_golden.py
```

## Not covered yet

- reading a training dataset ROOT file (`load_training_dataset`), to be written with the v3 dataset module,
- the command-line tools in `garf/bin`,
- `GarfDetector` end to end (initialisation, several heads, `save_projections`),
- GPU (CUDA, MPS) paths,
- analog vs ARF on a real detector: needs GATE, planned in the opengate tests.
