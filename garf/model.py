# -*- coding: utf-8 -*-
"""
ARF models: detection probabilities per channel from (direction, energy).

Channel 0 is "not detected", channels 1..n are the energy windows.

File format (.garf): safetensors, with the metadata as a JSON string under
the "garf" key of the header and the tensors named by role (x_mean, x_std,
net<k>.<layer>.weight|bias). Legacy .pth files (garf <= 2.11) can be read.
"""

import json
import warnings
from pathlib import Path

import torch

from .arrays import check_same_length, is_numpy, to_output, to_tensor, move
from .features import check_features, compute_features, resolve_feature_params

FORMAT = "garf-arf"
FORMAT_VERSION = 1
FILE_EXTENSION = ".garf"


def garf_version():
    try:
        from importlib.metadata import version

        return version("garf")
    except Exception:
        return "unknown"


def resolve_device(device="auto"):
    """'auto' (cuda, then mps, then cpu), 'cpu', 'gpu' (cuda or mps), 'cuda', 'mps' or a torch.device."""
    if isinstance(device, torch.device):
        return device
    if device in ("auto", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if device == "gpu":
            raise RuntimeError("garf: gpu requested but no cuda or mps device found")
        return torch.device("cpu")
    if device in ("cpu", "cuda", "mps"):
        return torch.device(device)
    raise ValueError(
        f"garf: unknown device '{device}', use 'auto', 'cpu', 'gpu', 'cuda' or 'mps'"
    )


class Mlp(torch.nn.Module):
    """
    Fully connected network: n_in -> H (ReLU) -> L x [H (ReLU)] -> n_out.
    Layer names (fc1, fcts.<i>, fc3) are those of garf <= 2.11 (Net_v1),
    so that legacy states can be loaded.
    """

    def __init__(self, n_in, H, L, n_out, activation="relu"):
        super().__init__()
        if activation != "relu":
            raise ValueError(f"garf: unsupported activation '{activation}'")
        self.fc1 = torch.nn.Linear(n_in, H)
        self.fcts = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(L)])
        self.fc3 = torch.nn.Linear(H, n_out)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0)
        for f in self.fcts:
            x = torch.clamp(f(x), min=0)
        return self.fc3(x)


class ResidualBlock(torch.nn.Module):
    """x -> relu(x + linear2(relu(linear1(x))))"""

    def __init__(self, H):
        super().__init__()
        self.linear1 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, H)

    def forward(self, x):
        r = self.linear2(torch.clamp(self.linear1(x), min=0))
        return torch.clamp(x + r, min=0)


class ResMlp(torch.nn.Module):
    """
    n_in -> H (ReLU) -> L residual blocks (2 layers each) -> n_out.
    Layer names (fc1, residual_layers.<i>.linear1/2, output_layer) are those
    of ResNet_v2 (garf 'models' branch, 2025), so that its states can be loaded.
    """

    def __init__(self, n_in, H, L, n_out, activation="relu"):
        super().__init__()
        if activation != "relu":
            raise ValueError(f"garf: unsupported activation '{activation}'")
        self.fc1 = torch.nn.Linear(n_in, H)
        self.residual_layers = torch.nn.ModuleList([ResidualBlock(H) for _ in range(L)])
        self.output_layer = torch.nn.Linear(H, n_out)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0)
        for layer in self.residual_layers:
            x = layer(x)
        return self.output_layer(x)


BLOCKS = ("mlp", "residual")


def apply_russian_roulette(probs, rr):
    """The training dataset kept 1/rr of the non-detected photons: undo it."""
    if rr == 1:
        return probs
    probs = probs.clone()
    probs[:, 0] *= rr
    return probs / probs.sum(dim=1, keepdim=True)


class ArfModel:
    """
    Base class (network or table). Subclasses implement _probs(x) and the
    tensors to save. Use ArfModel.load() to read a file.
    """

    model_type = None

    def __init__(self, features, n_channels, rr, metadata=None, feature_params=None):
        check_features(features)
        self.features = features
        self.feature_params = resolve_feature_params(features, feature_params)
        self.n_channels = int(n_channels)
        self.rr = float(rr)
        self.metadata = dict(metadata or {})
        self.device = torch.device("cpu")

    @property
    def n_windows(self):
        return self.n_channels - 1

    @property
    def distance_to_crystal(self):
        return self.metadata.get("detector", {}).get("distance_to_crystal")

    def to(self, device):
        self.device = resolve_device(device)
        return self

    def predict(self, ux, uy, uz, E):
        """
        Probabilities (n, n_channels) per channel, Russian roulette applied,
        rows sum to 1. numpy inputs -> numpy float64; torch inputs -> torch
        (on the model device).
        """
        numpy_out = is_numpy(ux, uy, uz, E)
        ux = to_tensor(ux)
        uy, uz, E = (to_tensor(a, like=ux) for a in (uy, uz, E))
        check_same_length(ux=ux, uy=uy, uz=uz, E=E)
        with torch.no_grad():
            x = compute_features(self.features, ux, uy, uz, E, self.feature_params)
            probs = self._probs(x)
        if numpy_out:
            probs = probs.cpu().double()  # (mps has no float64)
        return to_output(probs, numpy_out)

    def _probs(self, x):
        raise NotImplementedError

    # ---------------------------------------------------------------- io

    def full_metadata(self):
        m = dict(self.metadata)
        m.update(
            format=FORMAT,
            format_version=FORMAT_VERSION,
            model_type=self.model_type,
            features=self.features,
            n_channels=self.n_channels,
            rr=self.rr,
        )
        if self.feature_params is not None:
            m["feature_params"] = self.feature_params
        m.setdefault("provenance", {})
        return m

    def _tensors(self):
        raise NotImplementedError

    def save(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        if path.suffix != FILE_EXTENSION:
            warnings.warn(
                f"garf: model file '{path}' does not end with {FILE_EXTENSION}"
            )
        tensors = {k: v.detach().cpu().contiguous() for k, v in self._tensors().items()}
        meta = json.dumps(self.full_metadata(), default=_json_default)
        save_file(tensors, str(path), metadata={"garf": meta})

    @staticmethod
    def load(path, device="auto", legacy_state_index=-1):
        """
        Read a .garf file, or a legacy .pth (garf <= 2.11, with a warning;
        legacy_state_index selects the stored state, default the last one,
        which is what garf <= 2.11 used).
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"garf: model file '{path}' not found")
        if path.suffix == ".pth":
            model = load_legacy_pth(path, state_index=legacy_state_index)
        else:
            model = load_garf(path)
        return model.to(device)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"garf: cannot store {type(o)} in the metadata")


class MlpArfModel(ArfModel):
    """
    Neural network ARF (one network, or an ensemble whose probabilities are
    averaged). Inputs are normalised with x_mean / x_std (float64).
    """

    model_type = "mlp"

    def __init__(
        self,
        nets,
        x_mean,
        x_std,
        features,
        n_channels,
        rr,
        architecture,
        metadata=None,
        feature_params=None,
    ):
        super().__init__(features, n_channels, rr, metadata, feature_params)
        if not isinstance(nets, (list, tuple)):
            nets = [nets]
        if len(nets) == 0:
            raise ValueError("garf: at least one network is needed")
        self.nets = list(nets)
        for n in self.nets:
            n.eval()
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float64)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float64)
        self.architecture = dict(architecture)
        if self.architecture["n_out"] != self.n_channels:
            raise ValueError(
                f"garf: network has {self.architecture['n_out']} outputs, "
                f"but n_channels is {self.n_channels}"
            )

    @property
    def n_ensemble(self):
        return len(self.nets)

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(self.device)
        return self

    def _probs(self, x):
        # normalisation in the input dtype (float64 for numpy inputs, as garf <= 2.11)
        mean = move(self.x_mean, x.device, x.dtype)
        std = move(self.x_std, x.device, x.dtype)
        x = move((x - mean) / std, self.device, torch.float32)
        probs = None
        for net in self.nets:
            p = apply_russian_roulette(torch.softmax(net(x), dim=1), self.rr)
            probs = p if probs is None else probs + p
        return probs / len(self.nets)

    def full_metadata(self):
        m = super().full_metadata()
        m.update(architecture=self.architecture, n_ensemble=self.n_ensemble)
        return m

    def _tensors(self):
        t = {"x_mean": self.x_mean, "x_std": self.x_std}
        for k, net in enumerate(self.nets):
            for name, v in net.state_dict().items():
                t[f"net{k}.{name}"] = v
        return t

    @staticmethod
    def build_net(architecture):
        """architecture: n_in, H, L, n_out, activation, block ('mlp' or 'residual')."""
        a = architecture
        block = a.get("block", "mlp")
        if block not in BLOCKS:
            raise ValueError(f"garf: unknown network block '{block}', must be one of {BLOCKS}")
        cls = Mlp if block == "mlp" else ResMlp
        return cls(a["n_in"], a["H"], a["L"], a["n_out"], a.get("activation", "relu"))


def read_garf_metadata(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        header = f.metadata() or {}
    if "garf" not in header:
        raise ValueError(f"garf: '{path}' is not a garf model file (no garf metadata)")
    meta = json.loads(header["garf"])
    if meta.get("format") != FORMAT:
        raise ValueError(
            f"garf: '{path}' has format '{meta.get('format')}', not {FORMAT}"
        )
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"garf: '{path}' has format version {meta.get('format_version')}, "
            f"this garf reads version {FORMAT_VERSION}"
        )
    return meta


def load_garf(path):
    from safetensors.torch import load_file

    meta = read_garf_metadata(path)
    tensors = load_file(str(path))
    model_type = meta.get("model_type")
    common = {
        k: meta.pop(k)
        for k in ("features", "n_channels", "rr", "feature_params")
        if k in meta
    }
    for k in ("format", "format_version", "model_type"):
        meta.pop(k, None)
    if model_type == "mlp":
        architecture = meta.pop("architecture")
        n = meta.pop("n_ensemble", 1)
        nets = []
        for k in range(n):
            net = MlpArfModel.build_net(architecture)
            prefix = f"net{k}."
            state = {
                name[len(prefix) :]: v
                for name, v in tensors.items()
                if name.startswith(prefix)
            }
            net.load_state_dict(state)
            nets.append(net)
        return MlpArfModel(
            nets,
            tensors["x_mean"],
            tensors["x_std"],
            architecture=architecture,
            metadata=meta,
            **common,
        )
    if model_type == "lut":
        from .lut import LutArfModel

        meta.pop("table_shape", None)
        return LutArfModel(
            tensors["probs"],
            [tensors["edges_theta"], tensors["edges_phi"], tensors["edges_E"]],
            tensors["counts"],
            tensors["counts0"],
            metadata=meta,
            **common,
        )
    raise ValueError(f"garf: unknown model type '{model_type}' in '{path}'")


# legacy .pth: model_type -> network block, angle_param -> features
LEGACY_MODEL_TYPES = {"Net_v1": "mlp", "ResNet_v2": "residual"}
LEGACY_ANGLE_PARAMS = {"acos": "legacy_acos", "atan2": "theta_phi"}


def load_legacy_pth(path, state_index=-1):
    """
    garf <= 2.11 .pth file (pickle: only for trusted files). The stored
    states are those of the training epochs kept by garf_train; garf <= 2.11
    always used the last one.

    Features: legacy_acos, except for the files of the garf 'models' branch
    (2025) with angle_param = 'atan2': theta_phi. That branch also had the
    ResNet_v2 network (block 'residual') and used the straight shift at
    inference (recorded in the metadata as provenance.inference_shift_mode).
    """
    warnings.warn(
        f"garf: '{path}' is a legacy .pth model (garf <= 2.11); "
        f"convert it with garf_convert to the {FILE_EXTENSION} format",
        stacklevel=3,
    )
    nn = torch.load(str(path), map_location="cpu", weights_only=False)
    md = nn["model_data"]
    states = nn["optim"]["model_state"]
    epochs = [d.get("epoch") for d in nn["optim"].get("data", [{}] * len(states))]
    state = states[state_index]
    n_channels = int(md["n_ene_win"])
    model_type = md.get("model_type", "Net_v1")
    if model_type not in LEGACY_MODEL_TYPES:
        raise ValueError(
            f"garf: '{path}' has model_type '{model_type}', not supported "
            f"(supported: {sorted(LEGACY_MODEL_TYPES)})"
        )
    angle_param = md.get("angle_param", "acos")
    if angle_param not in LEGACY_ANGLE_PARAMS:
        raise ValueError(f"garf: '{path}' has unknown angle_param '{angle_param}'")
    features = LEGACY_ANGLE_PARAMS[angle_param]
    architecture = dict(n_in=3, H=int(md["H"]), L=int(md["L"]), n_out=n_channels)
    architecture["activation"] = "relu"
    architecture["block"] = LEGACY_MODEL_TYPES[model_type]
    net = MlpArfModel.build_net(architecture)
    net.load_state_dict(state)
    rr = md["rr"] if "rr" in md else md["RR"]

    skip = {"x_mean", "x_std", "current_gpu_mode", "current_gpu_device", "loss_values"}
    training = {k: v for k, v in md.items() if k not in skip}
    training["stored_epochs"] = epochs
    training["selected_epoch"] = epochs[state_index]
    if "loss_values" in md:
        training["loss_values"] = md["loss_values"]
    provenance = {"legacy_pth": Path(path).name, "garf_version": "<= 2.11"}
    if angle_param == "atan2":
        provenance["garf_version"] = "models branch (2025)"
        provenance["inference_shift_mode"] = "straight"
    metadata = {"training": training, "provenance": provenance}
    return MlpArfModel(
        [net],
        md["x_mean"],
        md["x_std"],
        features=features,
        n_channels=n_channels,
        rr=rr,
        architecture=architecture,
        metadata=metadata,
    )
