# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
import copy
from tqdm import tqdm
from .garf_model import Net_v1
from .helpers import get_gpu_device


def print_nn_params(params):
    """
    Print parameters of neural network
    """
    for e in params:
        if e[0] != "#":
            if e != "loss_values":
                print(e + " : " + str(params[e]))
            else:
                print(e + " : " + str(len(params[e])) + " values")
    print("GPU CUDA available : ", torch.cuda.is_available())
    print("GPU MPS  available : ", torch.backends.mps.is_available())
    if "current_gpu_mode" in params:
        print("GPU current        : ", params["current_gpu_mode"])


def nn_prepare_data(x_train, y_train, params):
    """
    Prepare the data for training: normalisation (mean/std) and add information
    in the model_data information structure.
    """
    # initialization
    if "seed" in params:
        if params["seed"] != "auto":
            torch.manual_seed(params["seed"])

    # Data normalization
    print("Data normalization")
    N = len(x_train)
    x_mean = np.mean(x_train, 0)
    x_std = np.std(x_train, 0)
    x_train = (x_train - x_mean) / x_std

    # Prepare data to be saved (merge with param)
    model_data = dict()
    model_data["x_mean"] = x_mean
    model_data["x_std"] = x_std
    model_data["N"] = N

    # copy param except comments
    for i in params:
        if not i[0] == "#":
            model_data[i] = params[i]

    # Use pytorch default precision (float32)
    x_train = x_train.astype("float32")

    # return
    return x_train, y_train, model_data, N


def nn_get_optimiser(model_data, model):
    """
    Create the optimize (Adam + scheduler)
    """
    learning_rate = model_data["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # decreasing learning_rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=False, patience=50
    )

    return optimizer, scheduler


def train_nn(x_train, y_train, params):
    """
    Train the ARF neural network.

    x_train -- x samples (3 dimensions: theta, phi, E)
    y_train -- output probabilities vector for N energy windows
    params  -- dictionary of parameters and options

    params contains:
    - n_ene_win
    - batch_size
    - batch_per_epoch
    - epoch_store_every
    - H
    - L
    - epoch_max
    - early_stopping
    - gpu_mode : auto cpu gpu
    """

    # Initialization
    x_train, y_train, model_data, N = nn_prepare_data(x_train, y_train, params)

    # One-hot encoding
    print("One-hot encoding")
    y_vals, y_train = np.unique(y_train, return_inverse=True)
    n_ene_win = len(y_vals)
    print("Number of energy windows:", n_ene_win)
    model_data["n_ene_win"] = n_ene_win

    # Device type
    current_gpu_mode, current_gpu_device = get_gpu_device(params["gpu_mode"])
    model_data["current_gpu_mode"] = current_gpu_mode
    print(f"Device GPU type is {current_gpu_mode}")

    # Batch parameters
    batch_per_epoch = model_data["batch_per_epoch"]
    batch_size = model_data["batch_size"]
    epoch_store_every = model_data["epoch_store_every"]

    # DataLoader
    print("Data loader batch_size", batch_size)
    print("Data loader batch_per_epoch", batch_per_epoch)
    train_data2 = np.column_stack((x_train, y_train))
    if current_gpu_mode == "mps":
        print("With device mps (gpu), convert data to float32", train_data2.dtype)
        train_data2 = train_data2.astype(np.float32)

    train_loader2 = DataLoader(
        train_data2,
        batch_size=batch_size,
        num_workers=1,
        # pin_memory=True,
        # shuffle=True,  # if false ~20% faster, seems identical
        shuffle=False,  # if false ~20% faster, seems identical
        drop_last=True,
    )

    # Create the main NN
    H = model_data["H"]
    L = model_data["L"]
    model = Net_v1(H, L, n_ene_win)

    # Create the optimizer
    optimizer, scheduler = nn_get_optimiser(model_data, model)

    # Main loop initialization
    epoch_max = model_data["epoch_max"]
    early_stopping = model_data["early_stopping"]
    best_loss = np.Inf
    best_epoch = 0
    best_train_loss = np.Inf
    loss_values = np.zeros(epoch_max + 1)

    # Print parameters
    print_nn_params(model_data)

    # create main structures
    nn = dict()
    nn["model_data"] = model_data
    nn["optim"] = dict()
    nn["optim"]["model_state"] = []
    nn["optim"]["data"] = []
    previous_best = 9999
    best_epoch_index = 0

    # set the model to the device (cpu or gpu = cuda or mps)
    model.to(current_gpu_device)

    # Main loop
    print("\nStart learning ...")
    pbar = tqdm(total=epoch_max + 1, disable=not params["progress_bar"])
    epoch = 0
    stop = False
    while (not stop) and (epoch < epoch_max):
        # Train pass
        model.train()
        train_loss = 0.0
        n_samples_processed = 0

        # Loop on the data batch (batch_per_epoch times)
        for batch_idx, data in enumerate(train_loader2):
            x = data[:, 0:3]
            y = data[:, 3]
            X = Tensor(x.to(model.fc1.weight.dtype)).to(current_gpu_device)
            Y = Tensor(y).to(current_gpu_device).long()

            # Forward pass
            Y_out = model(X)

            # Compute expected loss
            # combines log_softmax and nll_loss in a single function
            loss = F.cross_entropy(Y_out, Y)

            # Backward pass
            loss.backward()

            # Parameter update (gradient descent)
            optimizer.step()
            optimizer.zero_grad()
            batch_size = X.shape[0]  # important with variable batch sizes
            train_loss += loss.data.item() * batch_size
            n_samples_processed += batch_size

            # Stop when batch_per_epoch is reach
            if batch_idx == params["batch_per_epoch"]:
                break
        # end for loop train_loader

        # end of train
        train_loss /= n_samples_processed
        if train_loss < best_train_loss * (1 - 1e-4):
            best_train_loss = train_loss
        mean_loss = train_loss

        loss_values[epoch] = mean_loss
        if mean_loss < best_loss * (1 - 1e-4):
            best_loss = mean_loss
            best_epoch = epoch
        elif epoch - best_epoch > early_stopping:
            tqdm.write(
                "{} epochs without improvement, early stop.".format(early_stopping)
            )
            stop = True

        # scheduler for learning rate
        scheduler.step(mean_loss)

        # FIXME WRONG
        # Check if need to print and store this epoch
        if best_train_loss < previous_best:
            tqdm.write("Epoch {} loss is {:.5f}".format(epoch, best_loss))
            previous_best = best_train_loss

        if (
            (epoch != 0 and epoch % epoch_store_every == 0)
            or stop
            or epoch >= epoch_max - 1
        ):
            optim_data = dict()
            print("Store weights", epoch)
            optim_data["epoch"] = epoch
            optim_data["train_loss"] = train_loss
            state = copy.deepcopy(model.state_dict())
            nn["optim"]["model_state"].append(state)
            nn["optim"]["data"].append(optim_data)
            best_epoch_index = epoch

        # update progress bar
        pbar.update(1)
        epoch = epoch + 1

    # end for loop
    print("Training done. Best = {:.5f} at epoch {:.0f}".format(best_loss, best_epoch))

    # prepare data to be saved
    model_data["loss_values"] = loss_values
    model_data["final_epoch"] = epoch
    model_data["best_epoch"] = best_epoch
    model_data["best_epoch_index"] = best_epoch_index
    model_data["best_loss"] = best_loss

    return nn
