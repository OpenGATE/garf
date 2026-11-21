# garf tests

The tests check the numerical behaviour of garf: geometry, network output, images and training. The first ones (golden, geometry, image, train) were written before the v3 refactor to pin the behaviour of garf 2.11; the v3 tests (`test_v3_*.py`) check the new API against the same golden data, so that any change of the numbers is either detected or intended. The tests of the garf 2.11 API will be removed with that API.

- 100 tests: 96 pass, 4 are known bugs of the garf 2.11 API (`xfail`), see below.
- CPU only, deterministic, about 17 s in total.
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
| `tests/test_known_bugs.py` | known bugs, `xfail(strict=True)` | 4 |
| `tests/test_v3_api.py` | v3 API: `ArfModel`, `ArfProjector`, `.garf` format, `garf_convert` | 27 |
| `tests/test_v3_features.py` | v3 network input features, angle conventions of old datasets | 7 |
| `tests/test_v3_dataset.py` | v3 training datasets: ROOT reading (v3 and old formats), metadata, split | 11 |
| `tests/test_v3_train.py` | v3 trainer on synthetic data | 12 |
| `tests/test_v3_detector.py` | v3 standalone `GarfDetector` against the garf 2.11 one | 9 |
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

## v3 API (`test_v3_api.py`)

The legacy model is read from the `.pth` and from its conversion to `.garf` (tests marked `[pth, garf]` run with both). The inputs are the physical quantities passed by GATE (`x, y, ux, uy, uz, E, w`); the features are `legacy_acos`, the shift is `legacy`. The results must equal the golden data of garf 2.11 (same tolerances).

| test | checks |
| --- | --- |
| `test_legacy_pth_metadata` | the `.pth` gives features `legacy_acos`, 3 channels, `RR = 50`, the architecture, the stored and selected epochs |
| `test_legacy_pth_warns` | reading a `.pth` warns that the format is deprecated |
| `test_legacy_state_selection[last, first]` | default: the last stored state (what garf ≤ 2.11 always used); another one with `legacy_state_index`. `best_epoch_index` is not used: it is an index in old garf versions and an epoch number in garf 2.11 |
| `test_predict_golden[pth, garf]` | `ArfModel.predict(ux, uy, uz, E)` (numpy) = golden probabilities, numpy float64 output |
| `test_predict_torch_input[pth, garf]` | same with torch inputs, torch output |
| `test_project_golden[pth, garf]` | `ArfProjector.project`: pixel indices identical to the golden ones (`ix = v`, `iy = u`), values = weighted probabilities of the windows |
| `test_accumulate_golden[pth, garf] × [hit slice or not]` | `ArfProjector.accumulate` into a numpy image = golden image; the squared image = sum of `(w·p)²` per pixel |
| `test_accumulate_torch_image[pth, garf]` | float32 torch inputs and image (as in `GarfDetector`): same totals per channel |
| `test_save_load_round_trip` | `.pth` → `.garf` → `.garf`: same features, `RR`, architecture, metadata, identical weights |
| `test_garf_file_is_not_pickle` | the `.garf` header (format, version, type, features) is read without loading the tensors; only one state is stored |
| `test_wrong_format_version` | an unknown format version is an error |
| `test_ensemble_averages_probabilities` | an ensemble of 2 networks gives the mean of their probabilities |
| `test_projector_distance_from_model` | `distance_to_crystal` taken from the model metadata when not given; warning when the given one differs |
| `test_projector_needs_a_distance` | error when neither the projector nor the model gives `distance_to_crystal` |
| `test_shift_modes` | `legacy`: shift `d·(ux, uy)`; `straight`: `d·(ux, uy)/uz` |
| `test_non_square_image` | 32 × 16 image: array `(channels, 16, 32)`, corners and centre in the right pixels |
| `test_wrong_image_shape` | an image with the wrong shape is an error |
| `test_inputs_must_have_the_same_length` | inputs of different lengths are an error |
| `test_garf_convert_cli` | `garf_convert x.pth x.garf -d 74.625`: the converted model stores the distance and gives the golden results |

## v3 features (`test_v3_features.py`)

| test | checks |
| --- | --- |
| `test_features_values` | `ux_uy_E`, `legacy_acos`, `theta_phi` columns from known directions |
| `test_unknown_features` | an unknown feature set is an error |
| `test_direction_from_angles_round_trip[legacy_acos, theta_phi]` | direction → angles → direction, for all azimuths and `uz` down to 0.01 |
| `test_wrong_convention_is_detected` | `theta_phi` angles read as `legacy_acos` (and the reverse) are rejected, with a message suggesting the other convention |
| `test_unknown_convention` | an unknown angle convention is an error |
| `test_ux_uy_close_to_legacy_near_normal_incidence` | near normal incidence (`uz > 0.995`), `90° - acos(u)` equals `u` (in degrees) within 0.02°: the default `ux_uy_E` carries the same information as `legacy_acos` up to an affine map |

## v3 training datasets (`test_v3_dataset.py`)

| test | checks |
| --- | --- |
| `test_write_read_round_trip[without, with weights]` | `write_training_dataset` → `load_training_dataset`: same columns, weights, metadata (`rr`, `n_channels`) |
| `test_old_dataset[legacy_acos, theta_phi] × [window, w]` | old datasets (`Theta`, `Phi`, window column named `window` or `w`): error without the angle convention; with it, the directions are recovered |
| `test_old_dataset_wrong_convention` | `theta_phi` angles read as `legacy_acos`: error |
| `test_not_a_training_dataset` | a ROOT file without the "ARF (training)" tree, or a missing file: error |
| `test_n_channels` | from the metadata if given (a window may have no entry), else `max(window) + 1`; inconsistent metadata: error |
| `test_split` | random train/validation split: sizes, no entry lost, reproducible with the seed, fraction 0 allowed |
| `test_sidecar_metadata` | metadata in `<name>.json` next to the ROOT file (`rr`, `detector`) |

## v3 trainer (`test_v3_train.py`)

The synthetic problem is the one of `test_train.py` (garf 2.11 trainer), with the directions computed from the angles.

| test | checks |
| --- | --- |
| `test_training_recovers_known_probabilities[ux_uy_E, legacy_acos, theta_phi]` | H = 32, L = 1, 40 epochs: mean error < 0.015, max < 0.09 (0.12 for `theta_phi`). Measured: mean 0.009–0.010, max 0.06–0.07, against mean 0.014, max 0.091 for the garf 2.11 trainer. The selected epoch has the lowest validation loss |
| `test_training_uses_the_whole_dataset` | the ordered dataset of the known bug of the garf 2.11 trainer: probability 0.5 recovered |
| `test_weights_equal_duplication` | particle weights in the loss: weight 3 on detected entries moves `p1` from 0.3 to 0.5625 |
| `test_russian_roulette` | dataset with `rr = 10` (half detected after the roulette): predicted `p1 = 1/11` |
| `test_rr_must_be_known_and_consistent` | `rr` missing, or different in the parameters and the dataset metadata: error |
| `test_ensemble` | 3 networks, seeds `seed`, `seed+1`, `seed+2`, saved and read back, same predictions |
| `test_early_stopping` | the training stops 5 epochs after the best validation loss |
| `test_unknown_parameter` | unknown parameters (e.g. the garf 2.11 `batch_per_epoch`) are an error |
| `test_save_load_trained_model` | features, `distance_to_crystal` (from the dataset metadata) and the dataset info are stored in the model |
| `test_garf_train_cli` | `garf_train params.json train.root m.garf` |

## v3 standalone detector (`test_v3_detector.py`)

Particles: 50 000, in a 50 mm cube around the origin, isotropic, 70 % at 140.5 keV. Three gantry angles (0°, 100°, 230°), plane at 300 mm with the Intevo-like rotation used by gaga-phsp, 64 × 64 × 4.8 mm image, legacy model.

| test | checks |
| --- | --- |
| `test_detector_equals_old_numpy_path[without, with hit slice]` | same images as the garf 2.11 numpy path, driven as gaga-phsp did (rtol 1e-5). The 2.11 path is run with the hit slice, then the slice is dropped: without it, garf 2.11 writes the windows in channels 0..n-2 while gaga-phsp dropped channel 0 (window 1 lost) |
| `test_detector_close_to_old_torch_path[without, with hit slice]` | same totals per angle and channel as the garf 2.11 torch path (float32, rtol 1e-3) |
| `test_numpy_and_torch_batches_and_chunks` | numpy or torch particles, one or several chunks (`batch_size`): same images |
| `test_non_square_image` | 64 × 32 image: shape `(angles, channels, 32, 64)` (broken in the 2.11 torch path) |
| `test_images_itk` | ITK images: size, spacing `(sx, sy, 1)`, centred origin |
| `test_wrong_batch_shape` | particles must be `(n, 7)` |
| `test_missing_parameter` | `initialize` without `image_size`: error |

## Known bugs (`test_known_bugs.py`)

Each test describes the **correct** behaviour and is marked `xfail(strict=True)`. Today they fail. When a bug is fixed, the test passes, strict mode reports it as an error (`XPASS(strict)`), and the marker must be removed: the test then stays as a normal regression test.

| test | bug | how it fails today |
| --- | --- | --- |
| `test_gate_column_layout_gives_golden_result` | GATE ≥ 10.0.3 passes `x, y, ux, uy, uz, E, w`, garf reads `x, y, θ, φ, E` (and the weights only with 6 columns) | 337 of 2000 points leave the image (shift of about `d` on both axes), and the others have wrong probabilities |
| `test_torch_image_equals_numpy_non_square[hit slice or not]` | torch image path: `x` checked against `size[1]`, flat index `y·shape[0] + x` | `IndexError` with a 32 × 16 image |
| `test_training_uses_the_whole_dataset` | no shuffle and the epoch stops after `batch_per_epoch` (+1) batches: only the first entries are used | ordered dataset (first half channel 0, second half channel 1): predicted probability of channel 1 is 0.003 instead of 0.5 |

## Golden data

`tests/generate_golden.py` writes `tests/data/golden_v034.npz`: the inputs (`in_x, in_y, in_ux, in_uy, in_uz, in_E, in_w`) and the outputs of the current code (`probs`, `u`, `v`, `w_pred`, `image`).

Regenerate only when a change of the numbers is intended, and say why in the commit:

```
python tests/generate_golden.py
```

## Not covered yet

- the plotting command-line tools in `garf/bin`,
- GPU (CUDA, MPS) paths,
- analog vs ARF on a real detector: needs GATE, planned in the opengate tests.
