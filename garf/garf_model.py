# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Net_v1(nn.Module):
    """
    Define the NN architecture (version 1)

    Input:
    - H:          nb of neurons per layer
    - n_ene_win:  nb of energy windows

    Description:
    - simple Linear, fully connected NN
    - two hidden layers
    - Input X dimension is 3: angle1, angle2, energy
    - Output Y dimension is n_ene_win (one-hot encoding)
    - activation function is ReLu
    """

    def __init__(self, H, L, n_ene_win):
        super(Net_v1, self).__init__()
        # Linear include Bias=True by default
        self.fc1 = nn.Linear(3, H)
        self.L = L
        self.fcts = nn.ModuleList()
        for i in range(L):
            self.fcts.append(nn.Linear(H, H))
        self.fc3 = nn.Linear(H, n_ene_win)

    def forward(self, X):
        X = self.fc1(X)  # first layer
        X = torch.clamp(X, min=0)  # relu
        for i in range(self.L):
            X = self.fcts[i](X)  # hidden layers
            X = torch.clamp(X, min=0)  # relu
        X = self.fc3(X)  # output layer
        return X


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers."""

    def __init__(self, H):
        super().__init__()
        self.linear1 = nn.Linear(H, H)
        self.linear2 = nn.Linear(H, H)

    def forward(self, x):
        # Calculate the residual
        residual = self.linear1(x)
        residual = torch.clamp(residual, min=0)  # ReLU
        residual = self.linear2(residual)

        # Add the input to the residual (the "skip connection")
        # and apply the final activation for this block
        out = x + residual
        out = torch.clamp(out, min=0)  # ReLU
        return out


class ResNet_v2(nn.Module):
    """A ResNet-style architecture for the ARF problem."""

    def __init__(self, H, L, n_ene_win):
        super().__init__()
        # Initial layer to project input from 3 dimensions to H dimensions
        self.fc1 = nn.Linear(3, H)

        # A series of residual blocks
        self.residual_layers = nn.ModuleList([ResidualBlock(H) for _ in range(L)])

        # Final output layer
        self.output_layer = nn.Linear(H, n_ene_win)

    def forward(self, x):
        # Pass through the input layer and apply first activation
        x = self.fc1(x)
        x = torch.clamp(x, min=0)  # ReLU

        # Pass through all the residual blocks
        for layer in self.residual_layers:
            x = layer(x)

        # Final prediction
        x = self.output_layer(x)
        return x


class MultiTask_v3(nn.Module):
    """
    A multi-task ResNet architecture for the ARF problem.
    It has two output heads: one for acceptance and one for energy window classification.
    """

    def __init__(self, H, L, n_energy_windows):
        super().__init__()
        # Shared backbone
        self.fc1 = nn.Linear(3, H)
        self.residual_layers = nn.ModuleList([ResidualBlock(H) for _ in range(L)])

        # Head 1: Predicts detection vs. non-detection (1 output logit)
        self.acceptance_head = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1)
        )

        # Head 2: Predicts which detected window (n_energy_windows-1 outputs)
        # We subtract 1 because we don't need to predict the "non-detected" class here.
        self.energy_head = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, n_energy_windows - 1)
        )

    def forward(self, x):
        # Pass through the shared backbone
        x = self.fc1(x)
        x = torch.clamp(x, min=0)  # ReLU
        for layer in self.residual_layers:
            x = layer(x)

        # Get predictions from each head
        acceptance_logit = self.acceptance_head(x)
        energy_logits = self.energy_head(x)

        return acceptance_logit, energy_logits


class MultiTask_v3_loss:

    def __init__(self, y_train, rr_factor, current_gpu_device):
        print("init")
        self.current_gpu_device = current_gpu_device

        # Get class weights for the acceptance loss (same hybrid method as before)
        class_counts = np.bincount(y_train)
        # Create binary weights: weight for non-detected (0) vs. detected (1)
        weight_0 = 1.0 / (class_counts[0] + 1e-9)
        weight_1 = 1.0 / (np.sum(class_counts[1:]) + 1e-9)
        self.acceptance_weights = torch.tensor(
            [weight_0, weight_1], dtype=torch.float
        ).to(current_gpu_device)
        if rr_factor > 1:
            self.acceptance_weights[0] *= rr_factor
        self.acceptance_weights /= torch.mean(self.acceptance_weights)
        print(f"Acceptance loss weights: {self.acceptance_weights.cpu().numpy()}")

    def loss(self, Y_out, Y_true):
        # Forward pass - model now returns two outputs
        acceptance_logit, energy_logits = Y_out

        # --- Custom Loss Calculation ---

        # 1. Acceptance Loss (Binary)
        # Create binary target: 0 if non-detected, 1 if detected
        Y_binary_true = (Y_true > 0).float().view(-1, 1)
        # Use BCEWithLogitsLoss which is numerically stable and takes class weights
        # We need to compute pos_weight for the binary case
        pos_weight = torch.tensor(
            [self.acceptance_weights[1] / self.acceptance_weights[0]],
            device=self.current_gpu_device,
        )
        acceptance_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss1 = acceptance_loss_fn(acceptance_logit, Y_binary_true)

        # 2. Energy Loss (Categorical, only for detected events)
        detected_mask = Y_true > 0
        if torch.sum(detected_mask) > 0:
            # We subtract 1 from the labels because energy_head has (n-1) outputs
            # e.g., window 1 -> class 0, window 2 -> class 1
            Y_energy_true = Y_true[detected_mask] - 1
            energy_logits_detected = energy_logits[detected_mask]
            loss2 = F.cross_entropy(energy_logits_detected, Y_energy_true)
        else:
            # If no detected events in this batch, loss is zero
            loss2 = 0.0

        # 3. Total Loss
        loss = loss1 + loss2  # You can weight these, e.g., loss1 + 0.5 * loss2

        return loss
