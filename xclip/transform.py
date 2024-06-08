'''
Different transformations that can be applied to the data.
'''
import os
import h5py 
import torch

from models import LinearLayer


def LinearTransformation(x, bias=True):
    '''
    Linear transformation training process
    '''
    model = LinearLayer(x, bias)


    return x