#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Functions used to perform a planet injection routine.
"""

#%%
import numpy as np
from matplotlib import pyplot as plt

#%%
def appendlist(fpath, x):
    """
    Description:
        Append numpy array or list x to an existing .txt file.
    Input:
        fpath - file path
        x - list or numpy array to append
    """
    
    # append to .txt file
    with open(fpath, "a") as myfile:
        row = ' '.join(str(e) for e in x)
        myfile.write(row + "\n")
        
def appendheader(fpath, header):
    """
    Description:
        Append numpy array or list x to an existing .txt file.
    Input:
        fpath - file path
        header - header of the file
    """
    
    # append to .txt file
    with open(fpath, "a") as myfile:
        myfile.write(header + "\n")
        
def createtxt(fpath):
    """
    Description:
        Create an empty .txt file
    Input:
        fpath - file path
    """
    
    # create empty file
    open(fpath, 'a').close()

def getsnr_obs(datacube, model, dw = 25, binsize = 8):
    """
    Description:
        Get the SNR from the injected datacube.
    Input:
        datacube - numpy array containing data
        model - numpy array, injected model
        dw - estimate noise in this area
        binsize - binsize to use for the planet signal
    Output:
        signal - injected exoplanet signal
        noise - photon-shot-noise estimate
        snr - SNR of the injected exoplanet signal
    """
    
    # check model center
    ind = int(np.argmax(model[0, :]))
    prof = np.nanmean(datacube, axis = 1)
    
    # get the signal
    lowlim = np.max([0, ind - int(binsize/2.)])
    uplim = np.min([ind + int(binsize/2.), datacube.shape[0]])
    signal = np.nanmean(prof[lowlim:uplim])
    signal = np.abs(signal) # signal is defined to be positive

    # move planet towards the center
    noise_region1 = np.nanstd(datacube[0:dw])
    noise_region2 = np.nanstd(datacube[-1-dw:-1])

    # use left, right or both regions for noise estimate such that
    # overlap between the area where the signal is determined is avoided
    if ind in range(0, dw + int(binsize/2.)):
        noise = noise_region2
    elif ind in range(datacube.shape[0]-dw-int(binsize/2.),
                         datacube.shape[0]):
        noise = noise_region1
    else:
        noise = np.sqrt(noise_region1**2 + noise_region2**2)
    noise /= np.sqrt(binsize) # correct for averaging over multiple bins
    
    # get snr
    snr = signal/noise
    
    return signal, noise, snr

def getsnr_obs(datacube, model, dw = 25, binsize = 8):
    """
    Description:
        Estimate the SNR from the observation.
    Input:
        datacube - 2D numpy array with dimension (radvel, # timeframe)
                   datacube of one observation
        model - 2D numpy array with dimension (# time frame, radvel),
                injected model
        dw - integer, width of the sides to use for the noise regions
        binsize - integer, width of the bin for which to determine the signal
    Output:
        signal - float, signal at model center within bin
        noise - float, estimate of the photon shot noise at the edges
        snr - float, signal-to-noise ratio
    """
    
    # check model center
    ind = int(np.argmax(model[0, :]))
    lowlim = np.max([0, ind - int(binsize/2.)])
    uplim = np.min([ind + int(binsize/2.), datacube.shape[0]])
    
    # get the signal
    signal = np.nanmean(datacube[lowlim:uplim, :])
    datacube[lowlim:uplim, :] = 3.

    # use the regions at the edges to estimate the noise
    region1 = datacube[0:dw, :].flatten()
    region2 = datacube[-1-dw:-1, :].flatten()
    
    # only use region if signal is not measured inside the region
    if ind in range(0, dw + int(binsize/2.)):
        noise = np.std(region2)
        
    elif ind in range(datacube.shape[0]-dw-int(binsize/2.),
                         datacube.shape[0]):
        noise = np.std(region1)
    else:
        noise = np.std(np.append(region1, region2))
            
    # correct for the fact that the signal is the mean of multiple values
    N = len(datacube[lowlim:uplim, :].flatten())
    noise /= np.sqrt(N)
    snr = signal/noise
    
    return signal, noise, snr

def getn_sp(datacube, model, obsid_dict, binsize = 8):
    """
    Description:
        Extract noise of the stellar pulsations from the datacube.
    Input:
        datacube - 2D numpy array with dimensions (radvel, # timeframe) with
                   corrected datacube containing all observations
        model - 2D numpy array with dimensions (# time frame, radvel) of the
                injected model
        obsid_dict - dictionary with (key, value) = (obsid,
                     indices in datacube)
        binsize - integer, width of the bins in which to measure the signal
    Output:
        noise - standard deviation of the signals measured at the same position
                of the injected model
    """
    
    # check model center
    ind = int(np.argmax(model[0, :]))
    lowlim = np.max([0, ind - int(binsize/2.)])
    uplim = np.min([ind + int(binsize/2.), datacube.shape[0]])
    
    # loop over all observations and measure the signal
    nobs = len(obsid_dict.keys())
    signals = np.zeros(nobs)
    for n, (obsid, obsind) in enumerate(obsid_dict.items()):
        signals[n] = np.nanmean(datacube[lowlim:uplim, obsind])
    noise = np.std(signals)
    
    return noise

def plot_result(datacube, model, residual, outputfolder):
    """
    Description:
        Plot the datacube before and after applying PCA and the model used.
    Input:
        datacube - 2D numpy array of datacube
        model - 2D numpy array of PCA model
        residual - 2D numpy array
    """
    
    vmin = np.min([np.min(datacube), np.min(model), np.min(residual)])
    vmax = np.max([np.max(datacube), np.max(model), np.max(residual)])
    
    fig, axes = plt.subplots(3, 1)
    plt.suptitle('Injection', y = 0.98, fontsize = 15)
    
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Data', fontsize = 12)
    axes[0].imshow(datacube, extent=[0,100,0,1], aspect=25, vmin=vmin,
        vmax=vmax)
    
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Injected data', fontsize = 12)
    axes[1].imshow(model, extent=[0,100,0,1], aspect=25, vmin=vmin,
        vmax=vmax)
    
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    axes[2].set_title('Corrected injected data', fontsize = 12)
    axes[2].imshow(residual, extent=[0,100,0,1], aspect=25, vmin=vmin,
        vmax=vmax)
    
    fig.add_subplot(axes[1]).annotate('Radial velocity',
               xy = (-0.04,0.5), xycoords='axes fraction', rotation = 90,
               va='center', ha='center', fontsize = 15)
    fig.add_subplot(axes[2]).annotate('# time frame',
               xy = (0.5,-0.175), xycoords='axes fraction',
               va='center', ha='center', fontsize = 15)
    
    plt.savefig(os.path.join(outputfolder, 'injresult.pdf'))
    plt.savefig(os.path.join(outputfolder, 'injresult.png'),
                dpi = 300)
    plt.close()