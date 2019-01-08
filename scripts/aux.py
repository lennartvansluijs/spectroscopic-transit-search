#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Library of usefull auxillary functions and variables
    used by multiple scripts.
"""
#%%
import os
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform

#%%
def bin2d(x, binsize):
    """
    Description:
        Bin 2D data to specified binsize.
    Input:
        x - 2D numpy array
        binsize - integer specifying binsize
    Output:
        xb - binned list or numpy array
    """
    
    # bin the rows of 2D array x by the binsize
    xb = np.mean(x.reshape(int(x.shape[0]/binsize), binsize, x.shape[1]),
               axis = 1)
    
    return xb

def bin1d(x, binsize):
    """
    Description:
        Bin 1D data to specified binsize.
    Input:
        x - 1D numpy array
        binsize - integer specifying binsize
    Output:
        xb - binned list or numpy array
    """
    
    # bin the rows of 1D array x by the binsize
    xb = np.mean(x.reshape(int(len(x)/binsize), binsize), axis = 1)
    
    return xb

def get_flist(dirname):
    """
    Description:
        Get a sorted list of all files in the directory
    Input:
        dirname - name of the input directory
    """
    
    # get file list and sort
    flist = [fname for fname in os.listdir(dirname) if \
             os.path.isfile(os.path.join(dirname, fname))]
    flist = sorted(flist)
    
    return flist

def findall_containing_string(list, string, inelement = False):
    """
    Description:
        Find all elements containing a substring and return those elements.
    Input:
        list - list object
        string - string to look for
        invert - if True, find all that do not contain string
    """
    
    # return updated list (not) containing the substring
    if inelement:
        return [element for element in list if string in element]
    else:
        return [element for element in list if string not in element]
    
def wfits(fname, im):
    """
    Description:
        Write image data to a .fits file.
    Input:
        fname - file name
        im - image data
    """
    
    # wfits - write im to file fname, automatically overwriting any old file
    hea = fits.PrimaryHDU(im)
    hea.writeto(fname, overwrite=True)
    
def cen_rot2(im, rot, dim_out, offset1=(0,0), offset2=(0,0), order=2):
    """
    Description:
        takes a cube of images im, and a set of rotation angles in rot,
        and translates the middle of the frame with a size dim_out to the middle of
        a new output frame with an additional rotation of rot.
    Input:
        im - image data
        rot - rotation angle
        dim_out - image output dimension
        offset1 - first offset
        offset2 - second offset
        order - used order
    dst - rotated image
        """
    
    # convert to radian
    a = rot * np.pi / 180.
    
    # make a rotation matrix
    transform=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

    # -0.5 is there for fencepost counting error
    c_in = np.array(offset1) - 0.5
    c_out = 0.5 * np.array(dim_out) - 0.5

    # c_out has to be pre-rotated to make offset correct
    offset = c_in - c_out.dot(transform) - np.array(offset2).dot(transform)
    
    # perform the transformation
    dst=affine_transform( \
        im,transform.T, order=order,offset=offset, \
        output_shape=dim_out, cval=np.nan)
    
    return(dst)
    
def discrete_cmap(N, base_cmap=None):
    """
    Description:
        Create an N-bin discrete colormap from the specified input map.
    Input:
        N - integer, number of colors to use
        base_cmap - base colormap to use
    """

    # get base of discrete cmap
    base = plt.cm.get_cmap(base_cmap, N)
    
    return base

#%%
"""
Initizalize tableau20 color scheme used by many of the scripts.
"""

# define color scheme   
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127),
             (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207),
             (158, 218, 229)]

# scale the RGB values to the [0, 1] range
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)