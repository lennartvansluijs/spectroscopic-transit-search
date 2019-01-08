#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:05:42 2018

@author: lennart
"""
#%%
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from aux import *
from scipy import interpolate
from matplotlib import animation

#%%
def translate_1d(y, x0 = 0.):
    """
    Description:
        Translate a function y(x) -> y(x-x0).
    Input:
        y - list of y-values y(x)
        x0 - offset in x-direction
    Ouput:
        yt - list of y-values y(x-x0)
    """
    
    # interpolate function using a translation
    x = np.arange(len(y))
    f = interpolate.interp1d(x-x0, y, kind = 'linear', bounds_error = False,
                             fill_value = 0.)
    
    # get translated function
    yt = f(x)
    
    return yt

def shear(t, data, const):
    """
    Description:
        Shear the Spectral Time Series data.
    Input:
        t - list of times
        data - data of observational epoch
        const - shearing constant to use
    """
    
    yc = int(data.shape[1]/2.)
    tc = t[yc]
    dt = t - tc
    offset = dt * const

    data_s = np.copy(data)
    for n in range(data.shape[1]):
        data_s[:,n] = translate_1d(data[:,n], x0 = offset[n])
        
    return data_s, offset

def add_border(im, w = 50):
    """
    Description:
        Add a border to an image.
    Input:
        im - 2D numpy array image
        w - int width of the border
    Output:
        im_b - bordered 2D numpy array image
    """
    
    # add boarder
    im_b = np.zeros((im.shape[0]+w, im.shape[1]))
    im_b[:] = 0.
    lb = int(im_b.shape[0]/2.)-int(im.shape[0]/2.)
    rb = lb + im.shape[0]
    im_b[lb:rb, :] = im
    
    return im_b

def reverse_shearing(data_obs, offsets):
    """
    Description:
        Reverse shifting of previous alignment and shearing.
    Input:
        data_obs - 2D numpy array containing observational epoch
        offsets - previous shearing offsets
    Ouput:
        data_obs - 2D numpy array containing STS data after reversing
    """
    
    # reverse shearing
    for m in range(data_obs.shape[1]):
        data_obs[:, m] = translate_1d(data_obs[:, m], x0 = -offsets[m])

    return data_obs