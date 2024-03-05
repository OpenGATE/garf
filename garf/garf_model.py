# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


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
