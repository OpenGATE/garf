# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
import copy
import itk
import time
from tqdm import tqdm
from .nn_model import Net_v1
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
    nn = torch.load(filename, map_location=torch.device("cpu"))
    model_data = nn["model_data"]

    # set to gpu mode
    current_gpu_mode, current_gpu_device = get_gpu_device("cpu")
    model_data["current_gpu_device"] = current_gpu_device

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


def cast_image_type(image, pixel_type):
    OutputImageType = itk.Image[itk.F, image.GetImageDimension()]
    castImageFilter = itk.CastImageFilter[type(image), OutputImageType].New()
    castImageFilter.SetInput(image)
    castImageFilter.Update()
    return castImageFilter.GetOutput()


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
    if not "current_gpu_device" in model_data:
        current_gpu_mode, current_gpu_device = get_gpu_device(gpu_mode="auto")
        model_data["current_gpu_device"] = current_gpu_device
    # print(f"GARF {model_data['current_gpu_device']}")  # FIXME
    device = model_data["current_gpu_device"]
    model.to(device)

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
    angles_rad = np.deg2rad(angles)
    cos_theta = np.cos(angles_rad[:, 0])
    cos_phi = np.cos(angles_rad[:, 1])

    #  see in Gate_NN_ARF_Actor, line "phi = acos(dir.x())/degree;"
    tx = length * cos_phi
    #  see in Gate_NN_ARF_Actor, line "theta = acos(dir.y())/degree;"
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


def remove_out_of_image_boundaries2(u, v, w_pred, size):
    """
    Remove values out of the images (<0 or > size)
    """
    index = np.where(v < 0)[0]
    index = np.append(index, np.where(u < 0)[0])
    index = np.append(index, np.where(v > size[0] - 1)[0])
    index = np.append(index, np.where(u > size[1] - 1)[0])
    v = np.delete(v, index)
    u = np.delete(u, index)
    w_pred = np.delete(w_pred, index, axis=0)
    # print('Remove points out of the image: {} values removed'.format(len(index)))
    return u, v, w_pred


def image_from_coordinates_add(img, u, v, w_pred, hit_slice=False):
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

    # create range array which goes along with the arrays returned by bincount
    # (see man for np.bincount)
    uv32Bins = np.arange(np.amax(uv32) + 1, dtype=np.int32)

    # this will generate many 32bit values corresponding to 16bit value pairs
    # lying outside the image -> see conditions below

    # generate 16bit view to convert back and reshape
    uv16Bins = uv32Bins.view(dtype=np.uint16)
    hs = int((uv16Bins.size / 2))
    uv16Bins = uv16Bins.reshape((hs, 2))

    # fill image using index broadcasting
    # Important: the >0 condition is to avoid outside elements.
    tiny = 0  ## FIXME
    for i in range(1, nb_ene):
        # sum up values for pixel coordinates which occur multiple times
        chx = np.bincount(uv32, weights=w_pred[:, i])
        img[i, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] += chx[chx > tiny]

    # Consider the hit slice ?
    if hit_slice:
        for i in range(1, nb_ene):
            chx = np.bincount(uv32)
            img[0, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] += chx[chx > tiny]


def arf_plane_init(garf_user_info, rotation_angle, n):
    # initial image vectors
    plane_U = np.array([1, 0, 0])
    plane_V = np.array([0, 1, 0])
    # get rotation from the angle
    r = rotation_angle * garf_user_info.plane_rotation

    # new image plane vectors
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of the
    # two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    plane_normal = np.array([plane_normal] * n)

    # axial is Z axis
    center = np.array([0, 0, garf_user_info.plane_distance])
    center = r.apply(center)
    plane_center = np.array([center] * n)

    plane = {
        "plane_U": plane_U,
        "plane_V": plane_V,
        "rotation": r.inv(),
        "plane_normal": plane_normal,
        "plane_center": plane_center,
    }

    return plane


def arf_plane_project(x, plane, image_plane_size_mm):
    """
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    """

    # n is the normal plane, duplicated n times
    n = plane["plane_normal"][0 : len(x)]

    # c0 is the center of the plane, duplicated n times
    c0 = plane["plane_center"][0 : len(x)]

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)
    r = plane["rotation"]  # [0: len(x)]

    # p is the set of points position generated by the GAN
    p = x[:, 1:4]  # FIXME indices of the position

    # u is the set of points direction generated by the GAN
    u = x[:, 4:7]  # FIXME indices of the direction

    # w is the set of vectors from all points to the plane center
    w = p - c0

    # project to plane
    # dot product : out = (x*y).sum(-1)
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    # http://geomalgorithms.com/a05-_intersect-1.html
    # https://github.com/pytorch/pytorch/issues/18027

    # dot product between normal plane (n) and direction (u)
    ndotu = (n * u).sum(-1)

    # dot product between normal plane and vector from plane to point (w)
    si = -(n * w).sum(-1) / ndotu

    # only positive (direction to the plane)
    mask = si > 0
    mu = u[mask]
    mx = x[mask]
    mp = p[mask]
    msi = si[mask]
    # print(f"Remove negative direction, remains {mnb}/{len(x)}")#FIXME

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = np.array([msi] * 3).T

    # intersection between point-direction and plane
    psi = mp + msi * mu

    # offset of the head
    psi = psi + c0[: len(psi)]

    # apply the inverse of the rotation
    psip = r.apply(psi)

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0] / 2.0
    sizey = image_plane_size_mm[1] / 2.0
    mask1 = psip[:, 0] < sizex
    mask2 = psip[:, 0] > -sizex
    mask3 = psip[:, 1] < sizey
    mask4 = psip[:, 1] > -sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    mu = mu[m]
    mx = mx[m]
    nb = len(psip)
    # print(f"Remove points that are out of detector, remains {nb}/{len(x)}") #FIXME

    # reshape results
    pu = psip[:, 0].reshape((nb, 1))  # u
    pv = psip[:, 1].reshape((nb, 1))  # v
    y = np.concatenate((pu, pv), axis=1)

    # rotate direction according to the plane
    mup = r.apply(mu)
    norm = np.linalg.norm(mup, axis=1, keepdims=True)
    mup = mup / norm
    dx = mup[:, 0]
    dy = mup[:, 1]

    # FIXME -> clip arcos -1;1 ?

    # convert direction into theta/phi
    # theta is acos(dy)
    # phi is acos(dx)
    theta = np.degrees(np.arccos(dy)).reshape((nb, 1))
    phi = np.degrees(np.arccos(dx)).reshape((nb, 1))
    y = np.concatenate((y, theta), axis=1)
    y = np.concatenate((y, phi), axis=1)

    # concat the E
    E = mx[:, 0].reshape((nb, 1))
    data = np.concatenate((y, E), axis=1)

    return data


def build_arf_image_from_projected_points(garf_user_info, px, image):
    """
    Create a SPECT image from points on the ARF plane.

    Parameters are:
    - px = are the list of points projected on the plane
    - image are the current image to update
    """

    # get some variable
    nn = garf_user_info.nn
    model = garf_user_info.model_data

    # Get mean/std from the NN
    model_data = nn["model_data"]

    # get the two angles and the energy
    ax = px[:, 2:5]  ## FIXME keys indexes

    # predict weights
    w_pred = nn_predict(model, model_data, ax)

    # Get the two first columns = coordinates
    cx = px[:, 0:2]

    # Take angle into account: consider position at collimator + half crystal
    angles = px[:, 2:4]
    t = compute_angle_offset(angles, garf_user_info.distance_to_crystal)
    cx = cx[:, 0:2]
    cx = cx + t

    # convert x,y into pixel
    # Consider coordinates + half_size of the image - center pixel offset, and
    # convert into pixel with spacing
    coord = (
        cx + garf_user_info.image_plane_hsize_mm - garf_user_info.image_hspacing
    ) / garf_user_info.image_spacing
    coord = np.around(coord).astype(int)
    v = coord[:, 0]
    u = coord[:, 1]
    u, v, w_pred = remove_out_of_image_boundaries2(
        u, v, w_pred, garf_user_info.image_size
    )

    # convert array of coordinates to img
    image_from_coordinates_add(image, u, v, w_pred, hit_slice=garf_user_info.hit_slice)

    return u.shape[0]
