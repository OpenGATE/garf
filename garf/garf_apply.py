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
        self.image_size = None
        self.image_spacing = None
        self.crystal_distance = None

        # other members
        self.current_gpu_mode = None
        self.current_gpu_device = None
        self.nn = None
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.rr = None
        self.nb_ene = None
        self.detector_planes = None
        self.image_size_mm = None

        # FIXME to comment
        self.output_image = None
        self.img_zeros = None
        self.nprojs = None

        # output
        self.output_projections_itk = None
        self.output_projections_array = None

    def __str__(self):
        s = f"garf user gpu mode: {self.gpu_mode}\n"
        s += f"garf current gpu mode: {self.current_gpu_mode}\n"
        s += f"garf batch size: {self.batch_size}\n"
        s += f"garf nb energy windows: {self.nb_ene}\n"
        s += f"garf RR: {self.rr}\n"
        s += f"garf image size pix: {self.image_size}\n"
        s += f"garf image spacing: {self.image_spacing} mm\n"
        s += f"garf image size mm: {self.image_size_mm} mm\n"
        s += f"garf crystal distance: {self.crystal_distance} mm\n"
        if self.detector_planes is not None:
            s += f"garf nb of detector angles: {len(self.detector_planes)}\n"
        else:
            s += f"garf nb of detector angles: not initialized\n"
        return s

    def initialize1(self):
        # gpu mode
        self.current_gpu_mode, self.current_gpu_device = get_gpu_device(self.gpu_mode)

        # int
        self.batch_size = int(self.batch_size)

        # load NN
        self.nn, self.model = load_nn(self.pth_filename, verbose=False)
        self.model = self.model.to(self.current_gpu_device)

        # normalization
        md = self.nn["model_data"]
        dev = self.current_gpu_device
        if self.current_gpu_mode == 'mps':
            # on OSX, with MPS GPU, the float must be 32 bits, not 64 (yet)
            self.x_mean = torch.tensor(md['x_mean'].astype(np.float32), device=dev)
            self.x_std = torch.tensor(md['x_std'].astype(np.float32), device=dev)
        else:
            self.x_mean = torch.tensor(md['x_mean'], device=dev)
            self.x_std = torch.tensor(md['x_std'], device=dev)

        # Russian roulette
        if 'rr' in md:
            self.rr = md['rr']
        if 'RR' in md:
            self.rr = md['RR']
        if self.rr is None:
            print(f'Error in GARF, no value "rr" for Russian Roulette found in {md}')
            exit(-1)

    def initialize2(self):
        # Number of energy windows
        md = self.nn["model_data"]
        self.nb_ene = md["n_ene_win"]

        # Image size
        self.image_size_mm = [self.image_size[0] * self.image_spacing[0],
                              self.image_size[1] * self.image_spacing[1]]
        self.nprojs = len(self.detector_planes)
        # -1  # minus one because ??? first channel is "all"
        t_image_size = [self.nprojs, self.nb_ene - 1, self.image_size[0], self.image_size[1]]
        print("t_image_size", t_image_size)

        """
        FIXME 
        store output as list of 3D images ? or mix projections ? in a single file ? 
        ftm, previous was separated images. keep it separated
        """

        # FIXME to comments
        self.img_zeros = torch.zeros((self.image_size[0], self.image_size[1])).to(self.current_gpu_device)
        self.output_image = torch.zeros(tuple(t_image_size)).to(self.current_gpu_device)

    def initialize_planes(self, center, rotations):
        print('init planes')
        # FIXME first with direct, then with rotation gantry
        # FIXME was DetectorPlane
        self.detector_planes = []
        for rotation in rotations:
            dp = GarfDetectorPlane(self, center, rotation)
            self.detector_planes.append(dp)

    def project_to_planes(self, fake):
        for i, detector_plane in enumerate(self.detector_planes):
            batch = detector_plane.intersection(fake)
            self.arf_build_image_from_projected_points_torch(batch, i)

    def arf_build_image_from_projected_points_torch(self, batch, i):
        """
        :param batch: is a list of points in a plane
        :param i: the projection number
        :return:
        """
        # See ARFActor::arf_build_image_from_projected_points
        px = batch.clone()

        # convert direction in angles
        deg = np.pi / 180
        px[:, 2] = torch.arccos(batch[:, 2]) / deg
        px[:, 3] = torch.arccos(batch[:, 3]) / deg

        # get the two angles and the energy
        ax = px[:, 2:5]

        # predict weights
        w_pred = self.nn_predict_torch(self.model, self.nn["model_data"], ax)

        # need size and spacing on gpu (as tensor)
        image_size_pixel = torch.tensor(self.image_size).view(1, 2).to(self.current_gpu_device)
        image_spacing = torch.tensor(self.image_spacing).view(1, 2).to(self.current_gpu_device)
        print(self.image_size)
        print(self.image_spacing)
        image_size_mm = image_size_pixel * image_spacing
        print(image_size_mm)

        # Get the two first columns = points coordinates
        cx = px[:, 0:2]

        # need compute_angle_offset ?  FIXME

        print(f'image_size_pixel', image_size_pixel)
        print(f'image_size_mm', image_size_mm)
        print(f'image_plane_spacing', image_spacing)

        # positions
        coord = (cx + (image_size_pixel - 1) * image_spacing / 2) / image_spacing  ## FIXME ICXICICICICICIC ????
        # coord = (cx + (image_size_mm/2) * image_spacing / 2) / image_spacing
        uv = torch.round(coord).to(int)

        # FIXME ???
        # uv, w_pred = remove_out_of_image_boundaries_torch(uv, w, self.image_size_pixel)

        # do nothing if there is no hit in the image
        if uv.shape[0] == 0:
            return

        """# PW
        temp = self.zeros.fill_(0)
        temp = self.image_from_coordinates(temp, vu, w[:, 2])
        self.output_image[i, :, :] = self.output_image[i, :, :] + temp
        # SW
        temp = self.zeros.fill_(0)
        temp = self.image_from_coordinates(temp, vu, w[:, 1])
        self.output_image[i + self.nprojs, :, :] = self.output_image[i + self.nprojs, :, :] + temp
        """

        print("nb ene", self.nb_ene)
        print(f'{uv=}')
        print(f'vu shape {uv.shape=}')
        print(f'w shape {w_pred.shape=}')
        print(f'{i=}')
        for proj_w in range(1, self.nb_ene):
            print('projw', proj_w)
            temp = self.img_zeros.fill_(0)
            temp = self.image_from_coordinates(temp, uv, w_pred[:, proj_w])
            self.output_image[i, proj_w - 1, :, :] = self.output_image[i, proj_w - 1, :, :] + temp

    def nn_predict_torch(self, model, model_data, x):
        """
        Apply the NN to predict y from x
        """

        # apply input model normalisation
        x = (x - self.x_mean) / self.x_std

        vx = x.float()

        # predict values
        vy_pred = model(vx)

        # normalize probabilities
        y_pred = vy_pred
        y_pred = self.normalize_logproba_torch(y_pred)
        y_pred = self.normalize_proba_with_russian_roulette_torch(y_pred, 0, self.rr)

        return y_pred

    def save_projections(self):
        print(f'Saving projections')

        # convert to itk image
        self.output_projections_array = self.output_image.cpu().numpy()
        self.output_projections_itk = []
        for i in self.output_projections_array:
            self.output_projections_itk.append(itk.image_from_array(i))

        print(self.image_size)
        print('proj array shape', self.output_projections_array.shape)
        # print(self.output_projections_itk)

        # set spacing and origin like DigitizerProjectionActor
        spacing = self.image_spacing
        spacing = np.array([spacing[0], spacing[1], 1])
        print(spacing)
        size = np.array([0, 0, 0])
        size[0] = self.image_size[0]
        size[1] = self.image_size[1]  # FIXME WARNING order ??
        size[2] = self.output_projections_array.shape[0]
        origin = -size / 2.0 * spacing + spacing / 2.0
        origin[2] = 0
        print(f'Spacing {spacing}')
        print(f'Size {size}')
        print(f'origin {origin}')
        for i in range(len(self.output_projections_itk)):
            self.output_projections_itk[i].SetSpacing(spacing)
            self.output_projections_itk[i].SetOrigin(origin)
            itk.imwrite(self.output_projections_itk[i], f"toto_{i}.mhd")
            print(f'Output projection saved in : toto.mhd')

        # SC
        '''k = 0.5
        self.output_projections_SC_array = self.output_projections_array[:self.nprojs, :,
                                           :] - k * self.output_projections_array[self.nprojs:, :, :]
        self.output_projections_SC_array[self.output_projections_SC_array < 0] = 0
        self.output_projections_SC_itk = itk.image_from_array(self.output_projections_SC_array)
        size = np.array([256, 256, self.nprojs])
        origin = -size / 2.0 * spacing + spacing / 2.0
        self.output_projections_SC_itk.SetSpacing(spacing)
        self.output_projections_SC_itk.SetOrigin(origin)
        projs_SC_fn = self.output_fn.replace('.mhd', '_SC.mhd')
        itk.imwrite(self.output_projections_SC_itk, projs_SC_fn)
        print(f'Output projection (SC) saved in : {projs_SC_fn}')'''

        return self.output_projections_itk

    def normalize_logproba_torch(self, x):
        """
        Convert un-normalized log probabilities to normalized ones (0-100%)
        Not clear how to deal with exp overflow ?
        (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        """

        # exb = torch.exp(x)
        # exb_sum = torch.sum(exb, dim=1)
        # # divide if not equal at zero
        # p = torch.divide(exb.T, exb_sum,
        #               out=torch.zeros_like(exb.T)).T

        # check (should be equal to 1.0)
        # check = np.sum(p, axis=1)
        # print(check)

        b = x.amax(dim=1, keepdim=True)
        exb = torch.exp(x - b)
        exb_sum = torch.sum(exb, dim=1)
        p = torch.divide(exb.T, exb_sum, out=torch.zeros_like(exb.T)).T

        return p

    def normalize_proba_with_russian_roulette_torch(self, w_pred, channel, rr):
        """
        Consider rr times the values for the energy windows channel
        """
        # multiply column 'channel' by rr
        w_pred[:, channel] *= rr
        # normalize
        p_sum = torch.sum(w_pred, dim=1, keepdim=True)
        w_pred = w_pred / p_sum
        # check
        # p_sum = np.sum(w_pred, axis=1)
        # print(p_sum)
        return w_pred

    def image_from_coordinates(self, img, vu, w):
        img_r = img.ravel()
        print(f"{img.shape=}")
        print(f"{img_r.shape=}")
        ind_r = vu[:, 1] * img.shape[0] + vu[:, 0]
        # FIXME out of image ???
        if ind_r[0] > img.shape[0] * img.shape[1]:
            print(f'@@@@@@@@@@@@@@@@@@@@@@@ Error image? {ind_r}')
            print(f'{vu[:, 0]} {vu[:, 1]}')
            return img
        img_r.put_(index=ind_r, source=w, accumulate=True)
        img = img_r.reshape_as(img)
        return img


class GarfDetectorPlane:
    def __init__(self, garf_detector, center, rotation):
        self.garf_detector = garf_detector
        self.M = self.rotation_to_tensor(rotation)
        self.Mt = self.M.t()

        print('=====> Rotation in GarfDetectorPlane')
        print(f'{self.M=}')
        print(f'{self.Mt=}')
        print()

        dev = self.garf_detector.current_gpu_device
        print(f"center = {center}")
        center = torch.Tensor(center).float().to(dev)
        self.center = torch.matmul(self.M, center).to(dev)
        self.normal = -self.center / torch.norm(self.center)
        self.dd = torch.matmul(self.center, self.normal)

    def rotation_to_tensor(self, m):
        if self.garf_detector.current_gpu_mode == 'mps':
            print('MPS -> float32')
            m = m.astype(np.float32)
        t = torch.tensor([
            [m[0, 0], m[1, 0], m[2, 0]],
            [m[0, 1], m[1, 1], m[2, 1]],
            [m[0, 2], m[1, 2], m[2, 2]]]).float()  # unsure about float ?
        t = t.to(self.garf_detector.current_gpu_device)
        return t

    def intersection(self, batch):
        energ0 = batch[:, 0:1]
        pos0 = batch[:, 1:4]
        dir0 = batch[:, 4:7]

        # print(f' batch device {batch.device.type}')
        # print(f' dir0 device {dir0.device.type}')
        # self.normal = self.normal.to(self.garf_detector.current_gpu_device)
        # print(f' self.normal device {self.normal.device.type}')

        dir_dot_product = torch.sum(dir0 * self.normal, dim=1)
        t = (self.dd - torch.sum(pos0 * self.normal, dim=1)) / dir_dot_product

        '''# FIXME ? needed ? seems weirds
        mask = t > 0
        t = t[mask]
        dir0 = dir0[mask]
        pos0 = pos0[mask]
        dir_dot_product = dir_dot_product[mask]'''

        # distance from detector to the crystal
        dtc = self.garf_detector.crystal_distance
        print(f'crystal distance {dtc}')

        # position of the intersection
        pos_xyz = dir0 * t[:, None] + pos0
        pos_xyz_rot = torch.matmul(pos_xyz, self.Mt)  # .t())
        dir_xyz_rot = torch.matmul(dir0, self.Mt)  # .t())
        dir_xy_rot = dir_xyz_rot[:, 0:2]
        pos_xy_rot_crystal = pos_xyz_rot[:, 0:2] + dtc * dir_xy_rot  # FIXME

        # pos_xy_rot = torch.matmul(pos_xyz, self.Mt[[0,2], :].t()) # use this instead if the desired rotation axis is "z"
        # dir_xy_rot = torch.matmul(dir0, self.Mt[[0,2], :].t()) # use this instead if the desired rotation axis is "z"

        # pos_xyz_rot_crystal = pos_xyz_rot + (self.dist_to_crystal/dir_xyz_rot[:,2:3]) * dir_xyz_rot
        # pos_xy_rot_crystal = pos_xyz_rot_crystal[:,0:2]

        s = self.garf_detector.image_size_mm
        # print(f' pos xy {pos_xy_rot_crystal.shape}')
        # print(f' pos xy {pos_xy_rot_crystal.abs().shape}')
        # print(f' pos xy {pos_xy_rot_crystal.abs()[0].shape}')
        indexes_to_keep = torch.where((dir_dot_product < 0) &
                                      (t > 0) &  # only positive (direction to the plane)
                                      (pos_xy_rot_crystal[:, 0].abs() < s[0] / 2) &  # image size
                                      (pos_xy_rot_crystal[:, 1].abs() < s[1] / 2)
                                      )[0]

        batch_arf = torch.concat((pos_xy_rot_crystal[indexes_to_keep, :],
                                  dir_xy_rot[indexes_to_keep, :],
                                  energ0[indexes_to_keep, :]), dim=1)

        return batch_arf


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


def remove_out_of_image_boundaries_torch(vu, w_pred, image_size):
    """
    Remove values out of the images (<0 or > size)
    """
    len_0 = vu.shape[0]
    # index = torch.where((vu[:,0]>=0)
    #                     & (vu[:,1]>=0)
    #                     & (vu[:,0]< size[2])
    #                     & (vu[:,1]<size[1]))[0]
    # vu = vu[index]
    # w_pred = w_pred[index]

    vu_ = vu[(vu[:, 0] >= 0) & (vu[:, 1] >= 0) & (vu[:, 0] < image_size[1]) & (vu[:, 1] < image_size[0])]
    w_pred_ = w_pred[(vu[:, 0] >= 0) & (vu[:, 1] >= 0) & (vu[:, 0] < image_size[1]) & (vu[:, 1] < image_size[0])]

    if len_0 - vu.shape[0] > 0:
        print('Remove points out of the image: {} values removed sur {}'.format(len_0 - vu.shape[0], len_0))

    return vu_, w_pred_


def image_from_coordinates_add(img, u, v, w_pred, hit_slice=False):
    """
    Convert an array of pixel coordinates u,v (int) and corresponding weight
    into an image
    if hit_slice is True: the first slice stores all hits.
    If not, the first slice is empty
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
    tiny = 0  # FIXME
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

    print('====> Rotation arf_plane_init')
    print(f'{garf_user_info.plane_rotation.as_matrix()=}')
    print(f'{r.as_matrix()=}')

    # new image plane vectors
    plane_U = r.apply(plane_U)
    plane_V = r.apply(plane_V)

    # normal vector is the cross product of the
    # two direction vectors on the plane
    plane_normal = np.cross(plane_U, plane_V)
    print(f'{plane_normal=}')
    plane_normal = np.array([plane_normal] * nb)
    print(f'{plane_normal=}')
    print()

    # axial is Z axis
    center = np.array([0, 0, garf_user_info.plane_distance])
    center = r.apply(center)
    plane_center = np.array([center] * nb)

    print(f'garf_user_info.plane_distance={garf_user_info.plane_distance}')
    print(f'center={center}')

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


def arf_from_points_to_image_counts(px,  # 5D: 2 plane coordinates, 2 angles, 1 energy
                                    model,  # ARF neural network model
                                    model_data,  # associated model data
                                    distance_to_crystal,  # from detection plane to crystal center
                                    image_plane_size_mm,  # image plane in mm
                                    image_plane_size_pixel,  # image plane in pixel
                                    image_plane_spacing):  # image plane spacing
    """
    Input : position, direction on the detector plane, energy
    Compute
    - garf.nn_predict
    - garf.compute_angle_offset
    - garf.remove_out_of_image_boundaries2
    """
    # get the two angles and the energy
    ax = px[:, 2:5]

    # predict weights
    w_pred = nn_predict(model, model_data, ax)

    # Get the two first columns = points coordinates
    cx = px[:, 0:2]

    # Get the two next columns = angles
    angles = px[:, 2:4]

    # Take angle into account: consider position at collimator + half crystal
    t = compute_angle_offset(angles, distance_to_crystal)
    cx = cx + t

    # convert coord to pixel
    print(f'image_plane_size_mm', image_plane_size_mm)
    print(f'image_plane_spacing', image_plane_spacing)
    coord = (cx + image_plane_size_mm / 2 - image_plane_spacing / 2) / image_plane_spacing
    coord = np.around(coord).astype(int)
    v = coord[:, 0]
    u = coord[:, 1]

    # remove points outside the image
    u, v, w_pred = remove_out_of_image_boundaries2(u, v, w_pred, image_plane_size_pixel)

    return u, v, w_pred


def arf_build_image_from_projected_points_numpy(garf_user_info, px, image):
    """
    Create a SPECT image from points on the ARF plane.
    Parameters are:
    - px = are the list of points projected on the plane
    - image are the current image to update
    """

    # from projected points to image counts
    u, v, w_pred = arf_from_points_to_image_counts(px,
                                                   garf_user_info.model_data,
                                                   garf_user_info.nn["model_data"],
                                                   garf_user_info.distance_to_crystal,
                                                   garf_user_info.image_plane_size_mm,
                                                   garf_user_info.image_size,
                                                   garf_user_info.image_spacing)

    # convert array of coordinates to img
    image_from_coordinates_add(image, u, v, w_pred, hit_slice=garf_user_info.hit_slice)

    return u.shape[0]
