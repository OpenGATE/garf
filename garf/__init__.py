# garf <= 2.11 API (to be replaced by the v3 API below)
from .helpers import *
from .garf_train import *
from .garf_detector import *

# garf v3 API
from .features import (
    FEATURES,
    FEATURE_PARAMS,
    DEFAULT_FEATURES,
    resolve_feature_params,
    compute_features,
    direction_from_angles,
    check_angles,
)
from .model import (
    ArfModel,
    MlpArfModel,
    russian_roulette_of_theta,
    Mlp,
    resolve_device,
    FILE_EXTENSION,
    FORMAT_VERSION,
)
from .projector import ArfProjector, SHIFT_MODES
from .dataset import (
    DEFAULT_RR_THETA_EDGES,
    TrainingDataset,
    detection_probability_per_theta,
    load_training_dataset,
    optimal_russian_roulette,
    write_training_dataset,
)
from .trainer import TrainingParameters, train
from .symmetry import measure_symmetry, symmetry_table
from .lut import LutArfModel, build_lut
from .spectral import (
    SpectralArfModel,
    bin_window_fraction,
    deposit_bins_from_metadata,
    spectral_classes,
    window_response,
)
from .detector import GarfDetector  # replaces garf_detector.GarfDetector (2.11)
