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
    Mlp,
    resolve_device,
    FILE_EXTENSION,
    FORMAT_VERSION,
)
from .projector import ArfProjector, SHIFT_MODES
from .dataset import (
    TrainingDataset,
    load_training_dataset,
    write_training_dataset,
)
from .trainer import TrainingParameters, train
from .symmetry import measure_symmetry, symmetry_table
from .lut import LutArfModel, build_lut
from .detector import GarfDetector  # replaces garf_detector.GarfDetector (2.11)
