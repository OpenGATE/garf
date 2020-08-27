# -*- coding: utf-8 -*-
import numpy as np
import os
import uproot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import copy
import SimpleITK as sitk
import time
from tqdm import tqdm


# -----------------------------------------------------------------------------
def print_nn_params(params):
    '''
    Print parameters of neural network
    '''
    for e in params:
        if (e[0] != '#'):
            if (e != 'loss_values'):
                print(e+' : '+str(params[e]))
            else:
                print(e+' : '+str(len(params[e]))+' values')



# -----------------------------------------------------------------------------
def nn_prepare_data(x_train, y_train, params):
    '''
    Prepare the data fro training: normalisation (mean/std) and add informatino
    in the model_data information structure.
    '''
    # itialization
    torch.manual_seed(params['seed'])

    # Data normalization
    print("Data normalization")
    N = len(x_train)
    x_mean = np.mean(x_train, 0)
    x_std = np.std(x_train, 0)
    x_train = (x_train - x_mean) / x_std

    # Prepare data to be save (merge with param)
    model_data = dict()
    model_data['x_mean'] = x_mean
    model_data['x_std'] = x_std
    model_data['N'] = N

    # copy param except comments
    for i in params:
        if (not i[0] == '#'):
            model_data[i] = params[i]

    # Use pytorch default precision (float32)
    x_train = x_train.astype('float32')

    # return
    return x_train, y_train, model_data, N

# -----------------------------------------------------------------------------
def nn_init_cuda_type():
    '''
    CPU or GPU ?
    '''
    # CUDA or not CUDA ?
    gpu_mode = torch.cuda.is_available()
    dtypef = torch.FloatTensor
    dtypei = torch.LongTensor
    if (gpu_mode):
        print('CUDA GPU mode is ON')
        print('CUDA version:', torch.version.cuda)
        print('CUDA current device:', torch.cuda.current_device())
        print('CUDA device counts:', torch.cuda.device_count())
        print('CUDA device name:', torch.cuda.get_device_name(0))
        dtypef = torch.cuda.FloatTensor
        dtypei = torch.cuda.LongTensor
    return dtypef, dtypei, gpu_mode


# -----------------------------------------------------------------------------
def nn_get_optimiser(model_data, model):
    '''
    Create the optimize (Adam + scheduler)
    '''
    learning_rate = model_data['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # weight_decay=0.000001) ## Test

    # decreasing learning_rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           verbose=False, patience=50)

    return optimizer, scheduler


# -----------------------------------------------------------------------------
class Net_v1(nn.Module):
    '''
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
    '''

    def __init__(self, H, L, n_ene_win):
        super(Net_v1, self).__init__()
        self.fc1 = nn.Linear(3, H) # Linear include Bias=True by default
        self.L = L
        self.fcts = nn.ModuleList()
        for i in range(L):
            self.fcts.append(nn.Linear(H,H))
        self.fc3 = nn.Linear(H, n_ene_win)

    def forward(self, X):
        X = self.fc1(X) # first layer
        X = torch.clamp(X, min=0) # relu
        for i in range(self.L):
            X = self.fcts[i](X) # hidden layers
            X = torch.clamp(X, min=0) # relu
        X = self.fc3(X) # output layer
        return X


# -----------------------------------------------------------------------------
def train_nn(x_train, y_train, params):
    '''
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

    '''

    # Initialization
    x_train, y_train, model_data, N = nn_prepare_data(x_train, y_train, params)

    # One-hot encoding
    print("One-hot encoding")
    y_vals, y_train = np.unique(y_train, return_inverse=True)
    n_ene_win = len(y_vals)
    print("Number of energy windows:", n_ene_win)
    model_data['n_ene_win'] = n_ene_win

    # Set that Y are labels
    y_train = y_train.astype('int64')

    # CUDA type
    dtypef, dtypei, gpu_mode = nn_init_cuda_type()

    # Pytorch Dataset
    train_data = TensorDataset(torch.from_numpy(x_train).type(dtypef),
                               torch.from_numpy(y_train).type(dtypei))

    # Batch parameters
    batch_per_epoch = model_data['batch_per_epoch']
    batch_size = model_data['batch_size']
    epoch_store_every = model_data['epoch_store_every']

    # DataLoader
    print('Data loader batch_size', batch_size)
    print('Data loader batch_per_epoch', batch_per_epoch)
    train_data2 = np.column_stack((x_train, y_train))
    train_loader2 = DataLoader(train_data2,
                               batch_size=batch_size,
                               num_workers=1,
                               # pin_memory=True,
                               shuffle=True,  ## if false ~20% faster, seems identical
                               drop_last=True)

    # Create the main NN
    H = model_data['H']
    L = model_data['L']
    model = Net_v1(H, L, n_ene_win)

    # Create the optimizer
    optimizer, scheduler = nn_get_optimiser(model_data, model)

    # Main loop initialization
    epoch_max = model_data['epoch_max']
    early_stopping = model_data['early_stopping']
    best_loss = np.Inf
    best_epoch = 0
    best_train_loss = np.Inf
    percent = epoch_max/100.0
    loss_values = np.zeros(epoch_max+1)

    if (gpu_mode):
        print("Set model cuda ...")
        model.cuda()
    model_data['gpu_mode'] = gpu_mode

    # Print parameters
    print_nn_params(model_data)

    # create main structures
    nn = dict()
    nn['model_data'] = model_data
    nn['optim'] = dict()
    nn['optim']['model_state'] = []
    nn['optim']['data'] = []
    previous_best = 9999
    best_epoch_index = 0

    # Main loop
    print('\nStart learning ...')
    pbar = tqdm(total=epoch_max + 1, disable=not params['progress_bar'])
    epoch = 0
    stop = False
    while (not stop) and (epoch<epoch_max):

        # Train pass
        model.train()
        train_loss = 0.
        n_samples_processed = 0

        # Loop on the data batch (batch_per_epoch times)
        for batch_idx, data in enumerate(train_loader2):
            x = data[:,0:3]
            y = data[:,3]
            X, Y = Variable(x).type(dtypef), Variable(y).type(dtypei)

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
            if (batch_idx == params['batch_per_epoch']): break
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
            tqdm.write('{} epochs without improvement, early stop.'
                  .format(early_stopping))
            stop = True
            break

        # scheduler for learning rate
        scheduler.step(mean_loss)

        # FIXME WRONG
        # Check if need to print and store this epoch
        if (epoch % epoch_store_every == 0 or best_train_loss < previous_best):
            tqdm.write('Epoch {} best is {:.5f} at epoch {:.0f}'.
                       format(epoch, best_loss, best_epoch))
            optim_data = dict()
            optim_data['epoch'] = epoch
            optim_data['train_loss'] = train_loss
            state = copy.deepcopy(model.state_dict())
            nn['optim']['model_state'].append(state)
            nn['optim']['data'].append(optim_data)
            if best_train_loss < previous_best:
                best_epoch_index = len(nn['optim']['model_state'])-1
                previous_best = best_train_loss

        # update progress bar
        pbar.update(1)
        epoch = epoch + 1

    # end for loop
    print("Training done. Best = {:.5f} at epoch {:.0f}".format(best_loss, best_epoch))

    # prepare data to be save
    model_data['loss_values'] = loss_values
    model_data['final_epoch'] = epoch
    model_data['best_epoch'] = best_epoch
    model_data['best_epoch_index'] = best_epoch_index
    model_data['best_loss'] = best_loss

    return nn


# -----------------------------------------------------------------------------
def load_nn(filename, gpu='auto', verbose=True):
    '''
    Load a torch NN model + all associated info
    '''

    if gpu == 'auto':
        gpu_mode = torch.cuda.is_available()
    else:
        if not gpu:
            gpu_mode = False
        else:
            gpu_mode = True

    if (gpu_mode):
        print('Loading model {} with GPU'.format(filename))
        nn = torch.load(filename)
    else:
        print('Loading model {} *without* GPU'.format(filename))
        nn = torch.load(filename, map_location=lambda storage, loc: storage)
    model_data = nn['model_data']

    # FIXME
    verbose and print('nb stored ', len(nn['optim']['data']))
    for d in nn['optim']['data']:
        verbose and print(d['epoch'], d['train_loss'])

    if not 'best_epoch_eval' in model_data:
        best_epoch_eval = len(nn['optim']['data'])-1
    else:
        best_epoch_eval = model_data['best_epoch_index']

    # print('index, ', best_epoch_eval)

    # the best epoch is always the last stored
    # if (not 'best_epoch_eval' in model_data):
    #     best_epoch_eval = len(nn['optim']['data'])-1
    #     # print("(Warning no best_epoch_eval, using the last one)")
    # else:
    #     best_epoch_eval = model_data['best_epoch_eval']
    # best_epoch_eval = model_data['best_epoch_index']
    verbose and print('Index of best epoch = {}'.format(best_epoch_eval))
    verbose and print('Best epoch = {}'.format(nn['optim']['data'][best_epoch_eval]['epoch']))
    state = nn['optim']['model_state'][best_epoch_eval]
    H = model_data['H']
    n_ene_win = model_data['n_ene_win']
    L = model_data['L']
    model = Net_v1(H, L, n_ene_win)
    model.load_state_dict(state)
    return nn, model


# -----------------------------------------------------------------------------
def dump_histo(rmin, rmax, bins, x, filename):
    r = [rmin, rmax] # FIXME max ??? --> fction 
    histo, bin_edges = np.histogram(x, bins=bins, range=r, density=False)
    f = open(filename, 'w')
    for edge, hist in zip(bin_edges, histo):
        f.write(f'{edge} {hist}\n')
    f.close()


# -----------------------------------------------------------------------------
def build_arf_image_with_nn(nn, model, x, param, verbose=True, debug=False):
    '''
    Create the image from ARF simulation data and NN.
    Parameters are:
    - gpu_batch_size
    - size
    - spacing
    - length
    - N (nb of events for scaling)
    '''

    t1 = time.time()
    if verbose:
        print(param)

    # Get mean/std from the NN
    model_data = nn['model_data']
    x_mean = model_data['x_mean']
    x_std = model_data['x_std']
    rr = model_data['RR']
    verbose and print('rr',rr)

    #print(model_data)

    # Number of data samples
    N_detected = len(x)
    N_dataset = float(param['N_dataset'])
    N_scale = float(param['N_scale'])
    if verbose:
        print("Nb of events:          ", N_dataset)
        print("Nb of detected events: ", N_detected)
        print("N scale:               ", N_scale)

    # get the two angles and the energy
    ax = x[:, 2:5]

    if debug:
        #print(ax.shape)
        E = ax[:,2]
        theta = ax[:,0]
        phi = ax[:,1]
        b = 200
        dump_histo(0.0, 0.4, b, E, 'energy.txt')
        dump_histo(0.0, 180, b, theta, 'theta.txt')
        dump_histo(0.0, 180, b, phi, 'phi.txt')

    # loop by batch
    i = 0
    start_index = 0
    batch_size = param['gpu_batch_size']
    if N_detected <1:
        print('ERROR ? No detected count')
        exit(0)
    while start_index < N_detected:
        end = int(start_index+batch_size)
        if (end > N_detected):
            end = N_detected
        tx = ax[start_index:end]
        w = nn_predict(model, model_data, tx)
        if (i == 0):
            w_pred = w
        else:
            w_pred = np.vstack((w_pred, w))
        start_index = end
        if verbose:
            print("Generating counts: {}/{} ...".format(end, N_detected))
        i = i+1

    nb_ene = len(w_pred[0])


    # -----------------------------------------------------------------------------
    # Image parameters
    # image size in pixels
    size = [nb_ene, param['size'], param['size']]
    # image spacing in mm
    spacing = [param['spacing'], param['spacing'], 1]
    # collimator+ half crystal length in mm
    coll_l = param['length']

    if verbose:
        print('Image size', size)
        print('Image spacing ', spacing)
        print('Image detector length ', coll_l)

    # -----------------------------------------------------------------------------
    # Get the two first columns = coordinates
    cx = x[:, 0:2]

    # -----------------------------------------------------------------------------
    # consider image plane information
    psize = [size[1]*spacing[0], size[2]*spacing[1]]
    #print(psize)
    hsize = np.divide(psize, 2.0)

    # -----------------------------------------------------------------------------
    # Take angle into account: consider position at collimator + half crystal
    # length
    if verbose:
        print("Compute image positions ...")
    angles = x[:, 2:4]
    t = compute_angle_offset(angles, coll_l)
    # print('angle offset', t)
    cx = cx + t

    # -----------------------------------------------------------------------------
    # create outout image
    data_img = np.zeros(size, dtype=np.float64)

    # -----------------------------------------------------------------------------
    # convert x,y into pixel
    # Consider coordinates + half_size of the image - center pixel offset, and
    # convert into pixel with spacing
    offset = [spacing[0]/2.0, spacing[1]/2.0]
    coord = (cx+hsize-offset)/spacing[0:2]
    coord = np.around(coord).astype(int)
    v = coord[:, 0]
    u = coord[:, 1]
    u, v, w_pred = remove_out_of_image_boundaries(u, v, w_pred, size)

    if debug:
        b = 200
        dump_histo(0.0, 128, b, u, 'u.txt')
        dump_histo(0.0, 128, b, v, 'v.txt')
    
    # -----------------------------------------------------------------------------
    # convert array of coordinates to img
    if verbose:
        print("Channel 0 in the output image is set to zero, it CANNOT be compared to reference data")
        print("Compute image ", size, spacing, "...")
    data_img = image_from_coordinates(data_img, u, v, w_pred)

    # -----------------------------------------------------------------------------
    # write final image
    print('N_dataset', N_dataset)
    print('N_scale', N_scale)
    data_img = np.divide(data_img, N_dataset)
    data_img = np.multiply(data_img, N_scale)
    img = sitk.GetImageFromArray(data_img)
    origin = np.divide(spacing, 2.0)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img = sitk.Cast(img, sitk.sitkFloat32)
    if verbose:
        print("Computation time: {0:.3f} sec".format(time.time()-t1))

    # -----------------------------------------------------------------------------
    # also output the squared value
    w_pred = np.square(w_pred)    
    data_img = image_from_coordinates(data_img, u, v, w_pred)
    data_img = np.divide(data_img, N_dataset)
    data_img = np.multiply(data_img, N_scale)
    #data_img = data_img/(N_dataset**2)*(N_scale**2)
    sq_img = sitk.GetImageFromArray(data_img)
    sq_img.CopyInformation(img)
    sq_img = sitk.Cast(sq_img, sitk.sitkFloat32)

    return img, sq_img


# -----------------------------------------------------------------------------
def nn_predict(model, model_data, x):
    '''
    Apply the NN to predict y from x
    '''

    x_mean = model_data['x_mean']
    x_std = model_data['x_std']
    if ('rr' in model_data):
        rr = model_data['rr']
    else:
        rr = model_data['RR']

    # apply input model normalisation
    x = (x - x_mean) / x_std

    # gpu ?
    dtypef = torch.FloatTensor
    gpu_mode = torch.cuda.is_available()
    if (gpu_mode):
        dtypef = torch.cuda.FloatTensor
        model.cuda()

    # torch encapsulation
    x = x.astype('float32')
    vx = Variable(torch.from_numpy(x)).type(dtypef)

    # predict values
    vy_pred = model(vx)

    # convert to numpy and normalize probabilities
    if (gpu_mode):
        y_pred = vy_pred.data.cpu().numpy()
    else:
        y_pred = vy_pred.data.numpy()
    y_pred = y_pred.astype(np.float64)
    y_pred = normalize_logproba(y_pred)
    y_pred = normalize_proba_with_russian_roulette(y_pred, 0, rr)

    # return
    return y_pred


# -----------------------------------------------------------------------------
def compute_angle_offset(angles, length):
    '''
    compute the x,y offset according to the angle
    '''

    #max_theta = np.max(angles[:, 0])
    #min_theta = np.min(angles[:, 0])
    #max_phi = np.max(angles[:, 1])
    #min_phi = np.min(angles[:, 1])
    #print("min max theta {} {}".format(min_theta, max_theta))
    #print("min max phi {} {}".format(min_phi, max_phi))

    angles_rad = np.deg2rad(angles)
    cos_theta = np.cos(angles_rad[:, 0])
    cos_phi = np.cos(angles_rad[:, 1])

    tx = length * cos_phi    ## yes see in Gate_NN_ARF_Actor, line "phi = acos(dir.x())/degree;"
    ty = length * cos_theta  ## yes see in Gate_NN_ARF_Actor, line "theta = acos(dir.y())/degree;"
    t = np.column_stack((tx, ty))

    return t


# -----------------------------------------------------------------------------
def normalize_logproba(x):
    '''
    Convert un-normalized log probabilities to normalized ones (0-100%)
    Not clear how to deal with exp overflow ?
    (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
    '''
    exb = np.exp(x)
    exb_sum = np.sum(exb, axis=1)
    # divide if not equal at zero
    p = np.divide(exb.T, exb_sum,
                  out=np.zeros_like(exb.T),
                  where=exb_sum != 0).T
    # check (should be equal to 1.0)
    # check = np.sum(p, axis=1)
    # print(check)
    return p


# -----------------------------------------------------------------------------
def normalize_proba_with_russian_roulette(w_pred, channel, rr):
    '''
    Consider rr times the values for the energy windows channel
    '''
    # multiply column 'channel' by rr
    w_pred[:, channel] *= rr
    # normalize
    p_sum = np.sum(w_pred, axis=1, keepdims=True)
    w_pred = w_pred/p_sum
    # check
    # p_sum = np.sum(w_pred, axis=1)
    # print(p_sum)
    return w_pred


# -----------------------------------------------------------------------------
def remove_out_of_image_boundaries(u, v, w_pred, size):
    '''
    Remove values out of the images (<0 or > size)
    '''
    index = np.where(v < 0)[0]
    index = np.append(index, np.where(u < 0)[0])
    index = np.append(index, np.where(v > size[1]-1)[0])
    index = np.append(index, np.where(u > size[2]-1)[0])
    v = np.delete(v, index)
    u = np.delete(u, index)
    w_pred = np.delete(w_pred, index, axis=0)
    # print('Remove points out of the image: {} values removed'.format(len(index)))
    return u, v, w_pred


# -----------------------------------------------------------------------------
def image_from_coordinates(img, u, v, w_pred):
    '''
    Convert an array of pixel coordinates u,v (int) and corresponding weight
    into an image
    '''

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
    ch2 = [] # nb of hits in every pixel
    ones = np.ones_like(w_pred[:, 0])
    for i in range(1, nb_ene):
        a = np.bincount(uv32, weights=w_pred[:, i])
        b = np.bincount(uv32) ## FIXME this is optional
        ch.append(a)
        ch2.append(b)

    # init image
    img.fill(0.0)

    # create range array which goes along with the arrays returned by bincount
    # (see man for np.bincount)
    uv32Bins = np.arange(np.amax(uv32)+1, dtype=np.int32)

    # this will generate many 32bit values corresponding to 16bit value pairs
    # lying outside of the image -> see conditions below

    # generate 16bit view to convert back and reshape
    uv16Bins = uv32Bins.view(dtype=np.uint16)
    hs = int((uv16Bins.size/2))
    uv16Bins = uv16Bins.reshape((hs, 2))

    # fill image using index broadcasting
    # Important: the >0 condition is to avoid outside elements.
    tiny = 0  ## FIXME
    for i in range(1, nb_ene):
        chx = ch[i-1]
        img[i, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] = chx[chx > tiny]

    # first slice with position only
    for i in range(1, nb_ene):
        chx = ch2[i-1]
        img[0, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] += chx[chx > tiny]

    # end
    return img
