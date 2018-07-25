# -*- coding: utf-8 -*-
import numpy as np
import os
import uproot

def my_hello_world():
    '''
    Test
    '''
    print('My Hello world')



def my_hello_world2():
    '''
    Test
    '''
    print('TER My Hello world OKOKOOKK')


# -----------------------------------------------------------------------------
def open_training_dataset(filename):
    '''
    Open a training dataset in root format.
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
        print("This root file is not a PhaseSpace, keys are: ", f.keys())
        exit()
        data = f['ARF (training)']

    # Convert to arrays
    #print("PhaseSpace keys: ", data.keys())
    #data.show()
    a = data.arrays()
    theta = a[b'Theta']
    phi = a[b'Phi']
    E = a[b'E']
    w = a[b'w']

    return theta, phi, E, w
