# -*- coding: utf-8 -*-
import numpy as np
import os
import uproot
import torch


def load_training_dataset(filename):
    """
    Load a training dataset in root format (theta, phi, E, w)
    """
    # Check if file exist
    if not os.path.isfile(filename):
        print("File '" + filename + "' does not exist.")
        exit()

    # Check if this is a root file
    try:
        f = uproot.open(filename)
    except Exception:
        print("File '" + filename + "' cannot be opened, not a root file ?")
        exit()

    # Look for a single key named "ARF (training)"
    k = f.keys()
    try:
        data = f["ARF (training)"]
    except Exception:
        print("This root file is not a ARF (training), keys are: ", f.keys())
        exit()

    # Convert to arrays
    a = data.arrays(library="np")
    theta = a["Theta"]
    phi = a["Phi"]
    E = a["E"]
    if "w" in data.keys():
        w = a["w"]
    else:
        w = a["window"]
    data = np.column_stack((theta, phi, E, w))

    return data, theta, phi, E, w


def print_training_dataset_info(data, rr=40):
    """
    Print training dataset information
    """
    theta = data[:, 0]
    phi = data[:, 1]
    E = data[:, 2]
    w = data[:, 3]

    n = len(w)
    # outside window is zero
    n_nd = len(w[w == 0])
    print("Nb of particles:              {}".format(n))
    for i in range(6):
        print(f"Nb of particle in windows {i} : {len(w[w == i])}")
    print(f"RR for outside particle:     {rr}")
    print(f"Nb of particles (RR):         {n_nd * rr + n - n_nd}")
    print("Nb of non-detected particles: {} {:.2f}%".format(n_nd, n_nd / n * 100))
    print(
        "Nb of detected particles:     {} {:.2f}%".format(
            n - n_nd, (n - n_nd) / n * 100
        )
    )

    print("Min Max theta {:.2f} {:.2f} deg".format(np.amin(theta), np.amax(theta)))
    print("Min Max phi   {:.2f} {:.2f} deg".format(np.amin(phi), np.amax(phi)))
    print(
        "Min Max E     {:.2f} {:.2f} keV".format(np.amin(E * 1000), np.amax(E * 1000))
    )


def load_test_dataset(filename):
    """
    Load a test dataset in root format (theta, phi, E, x, y)
    """

    # Check if file exist
    if not os.path.isfile(filename):
        print("File '" + filename + "' does not exist.")
        exit()

    # Check if this is a root file
    try:
        f = uproot.open(filename)
    except Exception:
        print("File '" + filename + "' cannot be opened, not a root file ?")
        exit()

    # Look for a single key named "ARF (training)"
    k = f.keys()
    try:
        data = f["ARF (using)"]
    except Exception:
        print("This root file is not a ARF (testing), keys are: ", f.keys())
        exit()

    # Convert to arrays
    a = data.arrays()
    theta = a["Theta"]
    phi = a["Phi"]
    E = a["E"]
    x = a["X"]
    y = a["Y"]
    debug = True
    if debug:  # FIXME DEBUG
        evid = a["eventID"]
        data = np.column_stack((x, y, theta, phi, E, evid))
        print("DEBUG eventID")
    else:
        data = np.column_stack((x, y, theta, phi, E))

    return data, x, y, theta, phi, E


def image_uncertainty_arf(data, sq_data, N, threshold):
    """
    Compute image uncertainty for ARF image
    threshold as a % of the max
    """

    uncert = np.copy(data)
    uncert.fill(0.0)
    s_sigma = np.zeros(len(data))
    slice_i = 0
    while slice_i < len(data):
        # get slice for the ith energy windows
        d = data[slice_i]
        sq_d = sq_data[slice_i]

        # Chetty2007 p 4832 history by history
        a = np.divide(sq_d, N)
        b = np.square(np.divide(d, N))
        m = a - b
        m = m / (N - 1)
        sm = np.sqrt(m)

        # number of counts (per N)
        dn = np.divide(d, N)

        # normalisation by the nb of counts dn
        t = np.max(d) * threshold
        sigma = np.divide(sm, dn, out=np.zeros_like(dn), where=d > t)

        uncert[slice_i] = sigma

        # compute the mean over all pixels with more than threshold counts
        # n = np.where(d > threshold)
        n = np.where(d > t)
        n = len(n[0])
        sum_sigma = np.sum(sigma)
        if sum_sigma:
            mean_sigma = sum_sigma / n
        else:
            mean_sigma = 1

        s_sigma[slice_i] = mean_sigma
        slice_i = slice_i + 1

    return s_sigma, uncert


def image_uncertainty_analog(data, threshold):
    """
    Compute image uncertainty for image computed with analog MC
    threshold as a % of the max
    """

    uncert = np.copy(data)
    uncert.fill(0.0)
    s_sigma = np.zeros(len(data))
    slice_i = 0

    # loop
    while slice_i < len(data):
        d = data[slice_i]
        # sqrt of the variance (var is equal to the value itself)
        sigma = np.sqrt(d)
        t = np.max(d) * threshold
        sigma_n = np.divide(sigma, d, out=np.zeros_like(d), where=d > t)
        uncert[slice_i] = sigma_n
        sum_sigma = np.sum(sigma_n)
        # n_sigma = np.where(d > threshold) ## BUG !!
        n_sigma = np.where(d > t)
        n_sigma = len(n_sigma[0])
        mean_sigma = sum_sigma / n_sigma
        # end
        s_sigma[slice_i] = mean_sigma
        slice_i = slice_i + 1

    return s_sigma, uncert


def get_gpu_device(gpu_mode):
    current_gpu_mode = None
    current_gpu_device = None
    if gpu_mode == "cpu":
        current_gpu_device = torch.device("cpu")
        current_gpu_mode = "cpu"
        return current_gpu_mode, current_gpu_device

    if gpu_mode != "auto" and gpu_mode != "gpu":
        print(
            f'Error, gpu_mode can be : "cpu" or "gpu" or "auto", while it is {gpu_mode}'
        )
        exit(-1)

    if torch.backends.mps.is_available():
        current_gpu_device = torch.device("mps")
        current_gpu_mode = "mps"
    if torch.cuda.is_available():
        current_gpu_device = torch.device("cuda")
        current_gpu_mode = "cuda"

    if gpu_mode == "gpu" and current_gpu_mode is None:
        print("Error, no GPU on this device")
        print("")
        exit(-1)

    if gpu_mode == "auto" and current_gpu_mode is None:
        return get_gpu_device("cpu")

    return current_gpu_mode, current_gpu_device
