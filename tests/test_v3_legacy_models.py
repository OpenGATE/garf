# -*- coding: utf-8 -*-
"""
Legacy .pth variants: garf <= 2.11 (Net_v1, acos angles) and the garf
'models' branch (2025: angle_param 'atan2', ResNet_v2). Synthetic files: the
reference forward pass is written here independently of garf.
"""

import warnings

import numpy as np
import pytest
import torch

import garf


class RefResNetV2(torch.nn.Module):
    """ResNet_v2 of the garf 'models' branch (same layer names)."""

    def __init__(self, H, L, n_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, H)
        self.residual_layers = torch.nn.ModuleList()
        for _ in range(L):
            b = torch.nn.Module()
            b.linear1 = torch.nn.Linear(H, H)
            b.linear2 = torch.nn.Linear(H, H)
            self.residual_layers.append(b)
        self.output_layer = torch.nn.Linear(H, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for b in self.residual_layers:
            x = torch.relu(x + b.linear2(torch.relu(b.linear1(x))))
        return self.output_layer(x)


class RefNetV1(torch.nn.Module):
    def __init__(self, H, L, n_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, H)
        self.fcts = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(L)])
        self.fc3 = torch.nn.Linear(H, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for f in self.fcts:
            x = torch.relu(f(x))
        return self.fc3(x)


def write_pth(path, net, model_type, angle_param, x_mean, x_std, H, L, rr=50):
    md = dict(x_mean=x_mean, x_std=x_std, RR=rr, H=H, L=L, n_ene_win=3)
    if model_type is not None:
        md["model_type"] = model_type
    if angle_param is not None:
        md["angle_param"] = angle_param
    nn = dict(
        model_data=md, optim=dict(model_state=[net.state_dict()], data=[{"epoch": 7}])
    )
    torch.save(nn, path)


def directions(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u[:, 2] = np.abs(u[:, 2]) + 0.5
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, rng.uniform(0.11, 0.16, n)


def reference_probs(net, angles, E, x_mean, x_std, rr):
    x = (np.column_stack((angles[0], angles[1], E)) - x_mean) / x_std
    with torch.no_grad():
        logits = net(torch.from_numpy(x.astype("float32")))
        logits[:, 0] += np.log(rr)
        return torch.softmax(logits, dim=1).double().numpy()


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return garf.ArfModel.load(path, device="cpu")


@pytest.mark.parametrize(
    "ref_cls, model_type, block",
    [
        (RefNetV1, None, "mlp"),
        (RefNetV1, "Net_v1", "mlp"),
        (RefResNetV2, "ResNet_v2", "residual"),
    ],
)
@pytest.mark.parametrize("angle_param", [None, "acos", "atan2"])
def test_legacy_variants(tmp_path, ref_cls, model_type, block, angle_param):
    torch.manual_seed(0)
    H, L = 16, 2
    net = ref_cls(H, L, 3).eval()
    u, E = directions()
    if angle_param == "atan2":
        angles = (
            np.degrees(np.arccos(u[:, 2])),
            np.degrees(np.arctan2(u[:, 1], u[:, 0])),
        )
        x_mean, x_std = np.array([30.0, 0.0, 0.13]), np.array([17.0, 100.0, 0.02])
    else:
        angles = (np.degrees(np.arccos(u[:, 1])), np.degrees(np.arccos(u[:, 0])))
        x_mean, x_std = np.array([90.0, 90.0, 0.13]), np.array([30.0, 30.0, 0.02])
    f = tmp_path / "m.pth"
    write_pth(f, net, model_type, angle_param, x_mean, x_std, H, L)
    m = load(f)
    assert m.architecture["block"] == block
    assert m.features == ("theta_phi" if angle_param == "atan2" else "legacy_acos")
    if angle_param == "atan2":
        assert m.metadata["provenance"]["inference_shift_mode"] == "straight"
    expected = reference_probs(net, angles, E, x_mean, x_std, 50)
    p = m.predict(u[:, 0], u[:, 1], u[:, 2], E)
    np.testing.assert_allclose(p, expected, rtol=1e-5, atol=1e-7)

    # conversion to .garf keeps everything
    g = tmp_path / "m.garf"
    m.save(g)
    m2 = garf.ArfModel.load(g, device="cpu")
    assert m2.architecture["block"] == block and m2.features == m.features
    np.testing.assert_allclose(m2.predict(u[:, 0], u[:, 1], u[:, 2], E), p, rtol=1e-6)


def test_unsupported_legacy_model_type(tmp_path):
    net = RefNetV1(8, 1, 3)
    f = tmp_path / "m.pth"
    write_pth(f, net, "MultiTask_v4", "atan2", np.zeros(3), np.ones(3), 8, 1)
    with pytest.raises(ValueError, match="MultiTask_v4"):
        load(f)


def test_unknown_block():
    with pytest.raises(ValueError, match="block"):
        garf.TrainingParameters(block="transformer")
