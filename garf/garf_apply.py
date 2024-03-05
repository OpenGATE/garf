# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import Tensor
import itk
from .garf_model import Net_v1
from .helpers import get_gpu_device


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


class GarfDetector:

    def __init__(self):
        self.gpu_mode = "auto"
        self.pth_filename = None
        self.batch_size = 1e5
        self.image_size = [128, 128]
        self.image_spacing = [4.795, 4.795]

        # other members
        self.current_gpu_mode = None
        self.current_gpu_device = None

    def __str__(self):
        s = f"garf: user gpu mode: {self.gpu_mode}\n"
        s += f"garf: current gpu mode: {self.current_gpu_mode}"
        s += f"garf: batch size: {self.batch_size}"
        return s

    def initialize(self):
        # gpu mode
        self.current_gpu_mode, self.current_gpu_device = get_gpu_device(self.gpu_mode)
        # load NN
        FIXME HERE
        self.nn, self.model = garf.load_nn(self.pth_filename, verbose=False)
        self.model = self.model.to(self.device)


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


def arf_plane_init(garf_user_info, rotation, nb):
    print(f'rotation angle: {rotation}')
    # initial image vectors
    plane_U = np.array([1, 0, 0])
    plane_V = np.array([0, 1, 0])
    # get rotation from the angle
    r = rotation * garf_user_info.plane_rotation

    # new image plane vectors
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of the
    # two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    plane_normal = np.array([plane_normal] * nb)

    # axial is Z axis
    center = np.array([0, 0, garf_user_info.plane_distance])
    center = r.apply(center)
    plane_center = np.array([center] * nb)

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
    n = plane["plane_normal"][0: len(x)]

    # c0 is the center of the plane, duplicated n times
    c0 = plane["plane_center"][0: len(x)]

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

    # Get mean/std from the NN²
    model_data = nn["model_data"]

    # get the two angles and the energy
    ax = px[:, 2:5]  ## FIXME keys indexes

    # predict weights
    w_pred = nn_predict(model, model_data, ax)

    # Get the two first columns = coordinates²
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
