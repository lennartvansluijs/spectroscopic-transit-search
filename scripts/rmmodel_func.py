"""
Title: RM model

Authors: Ernst de Mooij & Lennart van Sluijs
Date: 23 Dec 2018

Description: Most of this code has been written by Ernst de Mooij to
             create Rossiter McLaughlin effected lineprofiles. Some small
             modifications were made by Lennart van Sluijs to utilize it for
             some more specific purposes.

Requirements:
    Requires C-code to be compiled using
    
    g++ -Wall -fPIC -O3 -march=native -fopenmp -c utils.cpp 
    g++ -shared -o libutils.so utils.o
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from numba import jit
import ctypes as c
from numpy.ctypeslib import ndpointer
import os
from matplotlib import gridspec

"""
This part was written by Ernst de Mooij.
"""

libUTILS = c.cdll.LoadLibrary('./libutils.so')

prototype = c.CFUNCTYPE(c.c_void_p,
    c.c_int,                
    c.c_int,                
    c.c_double,                
    c.c_double,                
    c.c_double,                
    ndpointer(c.c_double, flags="C_CONTIGUOUS")
)
make_planet_c = prototype(('make_planet', libUTILS))


prototype = c.CFUNCTYPE(c.c_void_p,
    c.c_int,                
    c.c_int,                
    c.c_double,                
    c.c_double,                
    ndpointer(c.c_double, flags="C_CONTIGUOUS")
)
dist_circle = prototype(('dist_circle', libUTILS))


prototype = c.CFUNCTYPE(c.c_void_p,
    c.c_int,                
    c.c_int,                
    c.c_double,                
    c.c_double,               
    c.c_double,               
    c.c_double,              
    c.c_double,                 
    ndpointer(c.c_double, flags="C_CONTIGUOUS")
)
make_star_c = prototype(('make_star', libUTILS))

@jit
def make_lineprofile(npix,rstar,xc,vgrid,A,veq,linewidth):
    """
    returns the line profile for the different points on the star
    as a 2d array with one axis being velocity and other axis position
    on the star
    npix - number of pixels along one axis of the star (assumes solid bosy rotation)
    rstar - the radius of the star in pixels
    xc - the midpoint of the star in pixels
    vgrid - the velocity grid for the spectrum you wish to make (1d array in km/s)
    A - the line depth of the intrinsic profile - the bottom is at (1 - A) is the max line depth (single value)
    veq - the equatorial velocity (the v sin i for star of inclination i) in km/s (single value)
    linewidth - the sigma of your Gaussian line profile in km/s (single value)
    """
    vc=(np.arange(npix)-xc)/rstar*veq
    vs=vgrid[np.newaxis,:]-vc[:,np.newaxis]
    profile=1.-A*np.exp( -(vs*vs)/2./linewidth**2)
    return profile

@jit
def make_star(npix0,osf,xc,yc,rstar,u1,u2,map0):
    """ 
    makes a circular disk with limb darkening
    returns a 2D map of the limb darkened star
    npix0 - number of pixels along one side of the output square array
    osf - the oversampling factor (integer multiplier)
    xc - x coordinate of the star
    yc - y coordinate of the star
    rstar - radius of the star in pixels
    u1 and u2 - linear and quadratic limb darkening coefficients
    """
    npix=int(np.floor(npix0*osf))
    make_star_c(npix,npix,xc*osf,yc*osf,rstar*osf,u1,u2,map0)
    star=map0.copy().reshape((npix0,osf,npix0,osf))
    star=star.mean(axis=3).mean(axis=1)
    return star


@jit
def make_planet(npix0,osf,xc,yc,rplanet,map0):
    """ 
    returns a 2D mask with the planet at 0.
    npix0 - number of pixels along one side of the output square array
    osf - the oversampling factor (integer multiplier)
    xc - x coordinate of the planet
    yc - y coordinate of the planet
    rplanet - radius of the planet in pixels
    """
    npix=int(np.floor(npix0*osf))
    make_planet_c(npix,npix,xc*osf,yc*osf,rplanet*osf,map0)
    planet=map0.copy().reshape((npix0,osf,npix0,osf))
    planet=planet.mean(axis=3).mean(axis=1)
    return planet

"""
This part was initally written by Ernst de Mooij, with some small modifications
by Lennart van Sluijs to utilitze the code for some specific purposes.
"""

def simimprof(radvel, A):

    ##initialise the model
    Rs_Rjup=17.9                                               #radius of the star in Rjup (rounded)
    Rp_Rjup=1.0                                                #radius of planet in Rjup
    Rs=510                                                     # radius of the star in pixels
    RpRs=Rp_Rjup/Rs_Rjup                                       # radius of planet in stellar radii
    Rp=RpRs*Rs                                                 # radius of the planet in stellar radii
    npix_star=1025                                             # number of pixels in the stellar map
    OSF=10                                                     # oversampling factor for star to avoid jagged edges
    map0=np.zeros((npix_star*OSF,npix_star*OSF))               # Map for calculating star, needed for C code
    u1=0.2752                                                  # linear limbdarkening coefficient
    u2=0.3790                                                  # quadratic limbdarkening coefficient
    xc=511.5                                                   # x-coordinate of disk centre
    yc=511.5                                                   # y-coordinate of disk centre
    veq=130.                                                   # V sin i (km/s)
    l_fwhm=20.                                                 # Intrinsic line FWHM (km/s)
    lw=l_fwhm/2.35                                             # Intinsice line width in sigma (km/s)
    
    vgrid=np.copy(radvel)                                      # velocity grid
    profile=make_lineprofile(npix_star,Rs,xc,vgrid,A,veq,lw)   # make line profile for each vertical slice of the star
    star=make_star(npix_star,OSF,xc,yc,Rs,u1,u2,map0)          # make a limb-darkened stellar disk
    sflux=star.sum(axis=1)                                     # sum up the stellar disk across the x-axis
    improf=np.sum(sflux[:,np.newaxis]*profile,axis=0)          # calculate the spectrum for an unocculted star
    improf=improf/improf[0]
    
    return improf

def simtransit(Rp_Rjup, radvel, A, b, theta, x0, outputfolder):

    ##initialise the model
    Rs_Rjup=17.9                                               #radius of the star in Rjup (rounded)
    Rs=510                                                     # radius of the star in pixels
    RpRs=Rp_Rjup/Rs_Rjup                                       # radius of planet in stellar radii
    Rp=RpRs*Rs                                                 # radius of the planet in stellar radii
    npix_star=1025                                             # number of pixels in the stellar map
    OSF=10                                                     # oversampling factor for star to avoid jagged edges
    map0=np.zeros((npix_star*OSF,npix_star*OSF))               # Map for calculating star, needed for C code
    u1=0.2752                                                  # linear limbdarkening coefficient
    u2=0.3790                                                  # quadratic limbdarkening coefficient
    xc=511.5                                                   # x-coordinate of disk centre
    yc=511.5                                                   # y-coordinate of disk centre
    veq=130.                                                   # V sin i (km/s)
    l_fwhm=20.                                                 # Intrinsic line FWHM (km/s)
    lw=l_fwhm/2.35                                             # Intinsice line width in sigma (km/s)
    
    vgrid=np.copy(radvel)                                      # velocity grid
    profile=make_lineprofile(npix_star,Rs,xc,vgrid,A,veq,lw)   # make line profile for each vertical slice of the star
    star=make_star(npix_star,OSF,xc,yc,Rs,u1,u2,map0)          # make a limb-darkened stellar disk
    sflux=star.sum(axis=1)                                     # sum up the stellar disk across the x-axis
    improf=np.sum(sflux[:,np.newaxis]*profile,axis=0)          # calculate the spectrum for an unocculted star
    normalisation=improf[0]
    improf=improf/normalisation

    #set up the orbit including a spin-orbit misalignment
    x0 = xc + x0 * Rs
    y0 = yc + np.zeros(len(x0)) + b * Rs
    x1=( (x0-xc)*np.cos(theta) - (y0-yc)*np.sin(theta)) + xc
    y1=( (x0-xc)*np.sin(theta) + (y0-yc)*np.cos(theta)) + yc
    
    #define some basic arrays
    line_profile1=np.zeros( (x1.shape[0],vgrid.shape[0]) )
    img1=np.zeros( (x1.shape[0],npix_star,npix_star) )
    
    #Loop over the positions of the planet
    for i in range(x1.shape[0]):    
        tmap1=star*make_planet(npix_star,OSF,y1[i],x1[i],Rp,map0)
        sflux=tmap1.sum(axis=0)
        line_profile1[i,:]=np.dot(sflux,profile)/normalisation
        img1[i,:,:]=tmap1
    
    #Calculate the differential profile as used for beta Pic
    diff_lineprof=line_profile1/((line_profile1[0,:])[np.newaxis,:])-1.
    diff_lineprof=diff_lineprof-(diff_lineprof[:,0])[:,np.newaxis]

    #write the arrays to disk
    hdu=fits.PrimaryHDU(img1)
    hdu.writeto(os.path.join(outputfolder, 'imgs.fits'),overwrite=True)
    hdu=fits.PrimaryHDU(line_profile1)
    hdu.writeto(os.path.join(outputfolder, 'lineprof.fits'),overwrite=True)
    hdu=fits.PrimaryHDU(diff_lineprof)
    hdu.writeto(os.path.join(outputfolder, 'diff_lineprof.fits'),
                overwrite=True)
    
    return diff_lineprof

def plot_injection(datacube, obsid_dict, RMM, outputfolder):
    """
    Description:
        Plot the spectral time series before injection, after injection and
        after injection + stellar pulsation removal.
    Input:
        STS - Spectral Time Series class for which to compare current state
              with inital state before injection and the injected model.
        RMM - RMModel class used for the injection.
        outputfolder - save plots here
    """
    fig, ax = plt.subplots()
    gs = gridspec.GridSpec(1, 3, wspace = 0.05, hspace = 0.05)
    ax1 = plt.subplot(gs[0, 0:1])
    ax2 = plt.subplot(gs[0, 1:2])
    ax3 = plt.subplot(gs[0, 2:3])
    
    vmax = np.max([np.max(RMM.obsdata), np.max(RMM.model),
                   np.max(datacube[:, obsid_dict[RMM.obsid]])])
    if vmax > 1e-5:
        vmin = 1e-5
    else:
        vmin = vmax/10.
    
    plt.suptitle('Transit injection', y = 0.8, fontsize = 15)
    
    ax1.imshow(RMM.obsdata,
              extent = [0,100,0,1], aspect = 100)
    ax1.annotate('data', xy = (0.5,1.1), xycoords='axes fraction',
    va='center', ha='center', color = 'k')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    im = ax2.imshow(RMM.model.swapaxes(0,1), extent = [0,100,0,1],
                    aspect = 100)
    ax2.annotate('model', xy = (0.5,1.1), xycoords='axes fraction',
    va='center', ha='center', color = 'k')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3.imshow(datacube[:, obsid_dict[RMM.obsid]],
              extent = [0,100,0,1], aspect = 100)
    ax3.annotate('injected data', xy = (0.5,1.1), xycoords='axes fraction',
    va='center', ha='center', color = 'k')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    plt.savefig(os.path.join(outputfolder, 'injection.pdf'))
    plt.savefig(os.path.join(outputfolder, 'injection.png'), dpi = 300)
    plt.close()