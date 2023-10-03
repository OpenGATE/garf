# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import copy
import itk
import time
from tqdm import tqdm
from .nn_model import Net_v1


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


def nn_prepare_data(x_train, y_train, params):
    """
    Prepare the data for training: normalisation (mean/std) and add informatino
    in the model_data information structure.
    """
    # initialization
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


def nn_init_device_type(gpu=True):
    """
    CPU or GPU ?
    """

    if gpu is False:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"

    print("Error, no GPU on this device")
    print("")
    exit(0)


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
    - gpu_mode
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
    device_type = nn_init_device_type(gpu=True)
    device = torch.device(device_type)
    model_data["device_type"] = device_type
    print(f"Device type is {device}")

    # Batch parameters
    batch_per_epoch = model_data["batch_per_epoch"]
    batch_size = model_data["batch_size"]
    epoch_store_every = model_data["epoch_store_every"]

    # DataLoader
    print("Data loader batch_size", batch_size)
    print("Data loader batch_per_epoch", batch_per_epoch)
    train_data2 = np.column_stack((x_train, y_train))
    if device_type == "mps":
        print("With device mps (gpu), convert data to float32", train_data2.dtype)
        train_data2 = train_data2.astype(np.float32)

    train_loader2 = DataLoader(
        train_data2,
        batch_size=batch_size,
        num_workers=1,
        # pin_memory=True,
        shuffle=True,  # if false ~20% faster, seems identical
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
    model.to(device)

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
            X = Tensor(x.to(model.fc1.weight.dtype)).to(device)
            Y = Tensor(y).to(device).long()

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
            break

        # scheduler for learning rate
        scheduler.step(mean_loss)

        # FIXME WRONG
        # Check if need to print and store this epoch
        if epoch % epoch_store_every == 0 or best_train_loss < previous_best:
            tqdm.write(
                "Epoch {} best is {:.5f} at epoch {:.0f}".format(
                    epoch, best_loss, best_epoch
                )
            )
            optim_data = dict()
            optim_data["epoch"] = epoch
            optim_data["train_loss"] = train_loss
            state = copy.deepcopy(model.state_dict())
            nn["optim"]["model_state"].append(state)
            nn["optim"]["data"].append(optim_data)
            if best_train_loss < previous_best:
                best_epoch_index = len(nn["optim"]["model_state"]) - 1
                previous_best = best_train_loss

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


def load_nn(filename, verbose=True):
    """
    Load a torch NN model + all associated info.
    Always load the model on cpu only.
    It should be moved to gpu only if needed.
    """

    verbose and print("Loading model ", filename)
    nn = torch.load(filename, map_location=torch.device('cpu'))
    model_data = nn['model_data']

    # set to cpu by default
    model_data["device"] = nn_init_device_type(False)

    # print some info
    verbose and print("nb stored ", len(nn["optim"]["data"]))
    for d in nn["optim"]["data"]:
        verbose and print(d["epoch"], d["train_loss"])

    # get the best epoch
    if not "best_epoch_eval" in model_data:
        best_epoch_eval = len(nn["optim"]["data"]) - 1
    else:
        best_epoch_eval = model_data["best_epoch_index"]
    verbose and print("Index of best epoch = {}".format(best_epoch_eval))
    verbose and print(
        "Best epoch = {}".format(nn["optim"]["data"][best_epoch_eval]["epoch"])
    )

    # prepare the model
    state = nn["optim"]["model_state"][best_epoch_eval]
    H = model_data["H"]
    n_ene_win = model_data["n_ene_win"]
    L = model_data["L"]
    model = Net_v1(H, L, n_ene_win)
    model.load_state_dict(state)

    return nn, model


def dump_histo(rmin, rmax, bins, x, filename):
    r = [rmin, rmax]  # FIXME max ??? --> fction
    histo, bin_edges = np.histogram(x, bins=bins, range=r, density=False)
    f = open(filename, "w")
    for edge, hist in zip(bin_edges, histo):
        f.write(f"{edge} {hist}\n")
    f.close()


def build_arf_image_with_nn(nn, model, x, param, verbose=True, debug=False):
    """
    Create the image from ARF simulation data and NN.
    Parameters are:
    - batch_size
    - size
    - spacing
    - length
    - N (nb of events for scaling)
    """

    t1 = time.time()
    if verbose:
        print(param)

    # Get mean/std from the NN
    model_data = nn["model_data"]
    x_mean = model_data["x_mean"]
    x_std = model_data["x_std"]
    rr = model_data["RR"]
    verbose and print("rr", rr)

    # print(model_data)

    # Number of data samples
    N_detected = len(x)
    N_dataset = float(param["N_dataset"])
    N_scale = float(param["N_scale"])
    if verbose:
        print("Nb of events:          ", N_dataset)
        print("Nb of detected events: ", N_detected)
        print("N scale:               ", N_scale)

    # get the two angles and the energy
    ax = x[:, 2:5]

    if debug:
        # print(ax.shape)
        E = ax[:, 2]
        theta = ax[:, 0]
        phi = ax[:, 1]
        b = 200
        dump_histo(0.0, 0.4, b, E, "energy.txt")
        dump_histo(0.0, 180, b, theta, "theta.txt")
        dump_histo(0.0, 180, b, phi, "phi.txt")

    # loop by batch
    i = 0
    start_index = 0
    batch_size = param["batch_size"]
    w_pred = None
    if N_detected < 1:
        print("ERROR ? No detected count")
        exit(0)
    while start_index < N_detected:
        end = int(start_index + batch_size)
        if end > N_detected:
            end = N_detected
        tx = ax[start_index:end]
        w = nn_predict(model, model_data, tx)
        if i == 0:
            w_pred = w
        else:
            w_pred = np.vstack((w_pred, w))
        start_index = end
        if verbose:
            print("Generating counts: {}/{} ...".format(end, N_detected))
        i = i + 1

    nb_ene = len(w_pred[0])

    # Image parameters
    # image size in pixels
    size = [nb_ene, param["size"], param["size"]]
    # image spacing in mm
    spacing = [param["spacing"], param["spacing"], 1]
    # collimator+ half crystal length in mm
    coll_l = param["length"]

    if verbose:
        print("Image size", size)
        print("Image spacing ", spacing)
        print("Image detector length ", coll_l)

    # Get the two first columns = coordinates
    cx = x[:, 0:2]

    # consider image plane information
    psize = [size[1] * spacing[0], size[2] * spacing[1]]
    hsize = np.divide(psize, 2.0)

    # Take angle into account: consider position at collimator + half crystal
    # length
    if verbose:
        print("Compute image positions ...")
    angles = x[:, 2:4]
    t = compute_angle_offset(angles, coll_l)
    cx = cx + t

    # create outout image
    data_img = np.zeros(size, dtype=np.float64)

    # convert x,y into pixel
    # Consider coordinates + half_size of the image - center pixel offset, and
    # convert into pixel with spacing
    offset = [spacing[0] / 2.0, spacing[1] / 2.0]
    coord = (cx + hsize - offset) / spacing[0:2]
    coord = np.around(coord).astype(int)
    v = coord[:, 0]
    u = coord[:, 1]
    u, v, w_pred = remove_out_of_image_boundaries(u, v, w_pred, size)

    if debug:
        b = 200
        dump_histo(0.0, 128, b, u, "u.txt")
        dump_histo(0.0, 128, b, v, "v.txt")

    # convert array of coordinates to img
    if verbose:
        print(
            "Channel 0 in the output image is set to zero, it CANNOT be compared to reference data"
        )
        print("Compute image ", size, spacing, "...")
    data_img = image_from_coordinates(data_img, u, v, w_pred)

    # write final image
    print("N_dataset", N_dataset)
    print("N_scale", N_scale)
    data_img = np.divide(data_img, N_dataset)
    data_img = np.multiply(data_img, N_scale)
    img = itk.GetImageFromArray(data_img)
    origin = np.divide(spacing, 2.0)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img = itk.Cast(img, itk.F)
    if verbose:
        print("Computation time: {0:.3f} sec".format(time.time() - t1))

    # also output the squared value
    w_pred = np.square(w_pred)
    data_img = image_from_coordinates(data_img, u, v, w_pred)
    data_img = np.divide(data_img, N_dataset)
    data_img = np.multiply(data_img, N_scale)
    # data_img = data_img/(N_dataset**2)*(N_scale**2)
    sq_img = itk.GetImageFromArray(data_img)
    sq_img.CopyInformation(img)
    sq_img = itk.Cast(sq_img, itk.F)

    return img, sq_img


def nn_predict(model, model_data, x):
    """
    Apply the NN to predict y from x
    GPU vs CPU is managed by the "device" variable in the mode_data dic
    WARNING : CPU is probably preferred here. This is a too small
    computation to really require GPU (which may prevent good multi-thread scalability)
    (Or maybe it is badly coded)
    """

    x_mean = model_data["x_mean"]
    x_std = model_data["x_std"]
    if "rr" in model_data:
        rr = model_data["rr"]
    else:
        rr = model_data["RR"]

    # apply input model normalisation
    x = (x - x_mean) / x_std

    # gpu ? (usually not)
    if not "device" in model_data:
        device_type = nn_init_device_type(gpu=False)
        device = torch.device(device_type)
        model_data["device"] = device
        model.to(device)
    device = model_data["device"]

    # torch encapsulation
    x = x.astype("float32")
    vx = Tensor(torch.from_numpy(x)).to(device)

    # predict values
    vy_pred = model(vx)

    # convert to numpy and normalize probabilities
    y_pred = normalize_logproba(vy_pred.data)
    y_pred = normalize_proba_with_russian_roulette(y_pred, 0, rr)
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred.astype(np.float64)

    # return
    return y_pred


def compute_angle_offset(angles, length):
    """
    compute the x,y offset according to the angle
    """

    # max_theta = np.max(angles[:, 0])
    # min_theta = np.min(angles[:, 0])
    # max_phi = np.max(angles[:, 1])
    # min_phi = np.min(angles[:, 1])
    # print("min max theta {} {}".format(min_theta, max_theta))
    # print("min max phi {} {}".format(min_phi, max_phi))

    angles_rad = np.deg2rad(angles)
    cos_theta = np.cos(angles_rad[:, 0])
    cos_phi = np.cos(angles_rad[:, 1])

    ## see in Gate_NN_ARF_Actor, line "phi = acos(dir.x())/degree;"
    tx = length * cos_phi
    ## see in Gate_NN_ARF_Actor, line "theta = acos(dir.y())/degree;"
    ty = length * cos_theta
    t = np.column_stack((tx, ty))

    return t


def normalize_logproba(x):
    """
    Convert un-normalized log probabilities to normalized ones (0-100%)
    Not clear how to deal with exp overflow ?
    (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
    """
    exb = torch.exp(x)
    exb_sum = torch.sum(exb, axis=1)
    # divide if not equal at zero
    p = torch.divide(exb.T, exb_sum, out=torch.zeros_like(exb.T)).T
    # check (should be equal to 1.0)
    # check = np.sum(p, axis=1)
    # print(check)
    return p


def normalize_proba_with_russian_roulette(w_pred, channel, rr):
    """
    Consider rr times the values for the energy windows channel
    """
    # multiply column 'channel' by rr
    w_pred[:, channel] *= rr
    # normalize
    p_sum = torch.sum(w_pred, axis=1, keepdims=True)
    w_pred = w_pred / p_sum
    # check
    # p_sum = torch.sum(w_pred, axis=1)
    # print(p_sum)
    return w_pred


def remove_out_of_image_boundaries(u, v, w_pred, size):
    """
    Remove values out of the images (<0 or > size)
    """
    index = np.where(v < 0)[0]
    index = np.append(index, np.where(u < 0)[0])
    index = np.append(index, np.where(v > size[1] - 1)[0])
    index = np.append(index, np.where(u > size[2] - 1)[0])
    v = np.delete(v, index)
    u = np.delete(u, index)
    w_pred = np.delete(w_pred, index, axis=0)
    # print('Remove points out of the image: {} values removed'.format(len(index)))
    return u, v, w_pred


def image_from_coordinates(img, u, v, w_pred):
    """
    Convert an array of pixel coordinates u,v (int) and corresponding weight
    into an image
    """

    # convert to int16
    u = u.astype(np.int16)
    v = v.astype(np.int16)

    # create a 32bit view of coordinate arrays to unite pairs of x,y into
    # single integer
    uv32 = np.vstack((u, v)).T.ravel().view(dtype=np.int32)

    # nb of energy windows
    nb_ene = len(w_pred[0])

    # sum up values for pixel coordinates which occur multiple times
    ch = []
    ch2 = []  # nb of hits in every pixel
    ones = np.ones_like(w_pred[:, 0])
    for i in range(1, nb_ene):
        a = np.bincount(uv32, weights=w_pred[:, i])
        b = np.bincount(uv32)  ## FIXME this is optional
        ch.append(a)
        ch2.append(b)

    # init image
    img.fill(0.0)

    # create range array which goes along with the arrays returned by bincount
    # (see man for np.bincount)
    uv32Bins = np.arange(np.amax(uv32) + 1, dtype=np.int32)

    # this will generate many 32bit values corresponding to 16bit value pairs
    # lying outside of the image -> see conditions below

    # generate 16bit view to convert back and reshape
    uv16Bins = uv32Bins.view(dtype=np.uint16)
    hs = int((uv16Bins.size / 2))
    uv16Bins = uv16Bins.reshape((hs, 2))

    # fill image using index broadcasting
    # Important: the >0 condition is to avoid outside elements.
    tiny = 0  ## FIXME
    for i in range(1, nb_ene):
        chx = ch[i - 1]
        img[i, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] = chx[chx > tiny]

    # first slice with position only
    for i in range(1, nb_ene):
        chx = ch2[i - 1]
        img[0, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] += chx[chx > tiny]

    # end
    return img
