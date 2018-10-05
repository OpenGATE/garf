# -*- coding: utf-8 -*-
import numpy as np
import os
import uproot
import copy

# -----------------------------------------------------------------------------
def load_training_dataset(filename):
    '''
    Load a training dataset in root format (theta, phi, E, w)
    '''
    # Check if file exist
    if (not os.path.isfile(filename)):
        print("File '"+filename+"' does not exist.")
        exit()

    # Check if this is a root file
    try:
        f = uproot.open(filename)
    except Exception:
        print("File '"+filename+"' cannot be opened, not a root file ?")
        exit()

    # Look for a single key named "ARF (training)"
    k = f.keys()
    try:
        data = f['ARF (training)']
    except Exception:
        print("This root file is not a ARF (training), keys are: ", f.keys())
        exit()

    # Convert to arrays
    a = data.arrays()
    theta = a[b'Theta']
    phi = a[b'Phi']
    E = a[b'E']
    w = a[b'window']
    data = np.column_stack((theta, phi, E, w))

    return data, theta, phi, E, w


# -----------------------------------------------------------------------------
def print_training_dataset_info(data, rr=40):
    '''
    Print training dataset information
    '''
    theta = data[:,0]
    phi = data[:,1]
    E = data[:,2]
    w = data[:,3]

    n = len(w)
    n_nd = len(w[w==0])
    print('Nb of particles:              {}'.format(n))
    print('Nb of particles (RR):         {}'.format(n_nd*rr+n-n_nd))
    print('Nb of non-detected particles: {} {:.2f}%'.format(n_nd, n_nd/n*100))
    print('Nb of detected particles:     {} {:.2f}%'.format(n-n_nd, (n-n_nd)/n*100))

    print('Min Max theta {:.2f} {:.2f} deg'.format(np.amin(theta), np.amax(theta)))
    print('Min Max phi   {:.2f} {:.2f} deg'.format(np.amin(phi), np.amax(phi)))
    print('Min Max E     {:.2f} {:.2f} keV'.format(np.amin(E*1000), np.amax(E*1000)))


# -----------------------------------------------------------------------------
def load_test_dataset(filename):
    '''
    Load a test dataset in root format (theta, phi, E, x, y)
    '''

    # Check if file exist
    if (not os.path.isfile(filename)):
        print("File '"+filename+"' does not exist.")
        exit()

    # Check if this is a root file
    try:
        f = uproot.open(filename)
    except Exception:
        print("File '"+filename+"' cannot be opened, not a root file ?")
        exit()

    # Look for a single key named "ARF (training)"
    k = f.keys()
    try:
        data = f['ARF (testing)']
    except Exception:
        print("This root file is not a ARF (testing), keys are: ", f.keys())
        exit()

    # Convert to arrays
    a = data.arrays()
    theta = a[b'Theta']
    phi = a[b'Phi']
    E = a[b'E']
    x = a[b'X']
    y = a[b'Y']
    data = np.column_stack((x, y, theta, phi, E))

    return data, x, y, theta, phi, E



