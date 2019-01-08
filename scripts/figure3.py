#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Oct 16 2018
Description:
    Run this to generate Figure 3a and Figure 3b.
"""

import os
import numpy as np
from astropy.io import fits
from sts_class import SpectralTimeSeries
from aux import *
from matplotlib import pyplot as plt
from rmmodel_func import *
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

# initialize from lineprofile cuts of the observational data
# by the UVES spectograph.
sts = SpectralTimeSeries()
sts.init_from_obs('../data', '../output/figure3/sts_init')
sts.remove_NaN()
sts.save('../output/figure3/sts_data', 'sts')

# load the RM-model
models = fits.getdata('../data/rm_models/diff_lineprof.fits')

# first candidate is 58004
obsid = '58004' # observation ID of epoch
dw = 10 # zoomed range to show [pixel]

# get datacube before and after stellar pulsation removal
sts_before = SpectralTimeSeries() 
sts_before.load('../output/figure3/sts_data', 'sts')
sts_after = SpectralTimeSeries() 
sts_after.load('../output/figure3/sts_data', 'sts')
sts_after.correct_for_pulsations()

# two candidates: obsid 58004 and 58098
obsids = ['58004', '58098']
fignames = ['figure3a', 'figure3b']
modelinds = [62, 34]
for n in range(2):
    
    # get the data from the epoch with the specified observation ID
    sts_obs_before = sts_before.datacube[:, sts.obsid_dict[obsids[n]]]
    sts_obs_after = sts_after.datacube[:, sts.obsid_dict[obsids[n]]]
    
    # get the best-fitting RM-model
    modelprof = models[modelinds[n], :]
    ind = np.argmax(modelprof)
    xx, sts_model = np.meshgrid(np.arange(sts_obs_after.shape[1]), modelprof)
    
    # get the residual
    sts_res = sts_obs_after - sts_model
    
    # get time stamps and convert to minutes
    time = sts.time[sts.obsid_dict[obsid]]
    time = (time - time[0]) * 24 * 60
    
    # setup grid for plot
    gs = gridspec.GridSpec(2, 8, width_ratios=[1.,1.,1.,1.,
                                               1.,1.,0.25,0.25])
    
    ax1 = plt.subplot(gs[0:1, 0:3])
    ax2 = plt.subplot(gs[0:1, 3:6])
    ax3 = plt.subplot(gs[1:2, 0:2])
    ax4 = plt.subplot(gs[1:2, 2:4])
    ax5 = plt.subplot(gs[1:2, 4:6])
    cbax1 = plt.subplot(gs[0:2, 6:7])
    
    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    
    # setup global colorbar range
    vmin = min(np.min(sts_obs_before), np.min(sts_obs_after))
    vmax = max(np.max(sts_obs_before), np.max(sts_obs_after))
    im1 = ax2.imshow(np.rot90(sts_obs_before, 1),
                     extent = [sts.radvel[0], sts.radvel[-1],
                       time[0], time[-1]],
                               aspect = 4.5 * abs(sts.radvel[ind-dw] \
                                                  - sts.radvel[ind+dw])/ \
                                                  (time[-1] - time[0]),
                     vmin = vmin, vmax = vmax, cmap = 'Greys_r')
    ax1.imshow(np.rot90(sts_obs_after, 1),
               extent = [sts.radvel[0], sts.radvel[-1],
                       time[0], time[-1]],
                         aspect = 4.5 * abs(sts.radvel[ind-dw] - \
                                            sts.radvel[ind+dw]) \
                                            /(time[-1] - time[0]),
                     vmin = vmin, vmax = vmax, cmap = 'Greys_r')
    ax3.imshow(np.rot90(sts_obs_after[ind-dw:ind+dw], 1),
               extent = [sts.radvel[ind-dw], sts.radvel[ind+dw],
                       time[0], time[-1]], aspect = 1.25 * \
    abs(sts.radvel[ind-dw] - sts.radvel[ind+dw])/(time[-1] - time[0]),
                     vmin = vmin, vmax = vmax, cmap = 'Greys_r')
    ax4.imshow(np.rot90(sts_model[ind-dw:ind+dw], 1),
               extent = [sts.radvel[ind-dw], sts.radvel[ind+dw],
                       0, 1],
                         aspect = 1.25 * abs(sts.radvel[ind-dw] - \
                                             sts.radvel[ind+dw]),
                     vmin = vmin, vmax = vmax, cmap = 'Greys_r')
    ax5.imshow(np.rot90(sts_res[ind-dw:ind+dw], 1),
               extent = [sts.radvel[ind-dw], sts.radvel[ind+dw],
                       0, 1],
                         aspect = 1.25 * \
                         abs(sts.radvel[ind-dw] - sts.radvel[ind+dw]),
                     vmin = vmin, vmax = vmax, cmap = 'Greys_r')
    
    # add titles and labels
    ax1.set_title('After pulsation removal', size = 12)
    ax2.set_title('Before pulsation removal', size = 12)
    ax4.set_title('Model', size = 12)
    ax5.set_title('Residual', size = 12)
    ax4.set_xlabel('Radial velocity [km/s]', size = 15)
    ax1.set_ylabel('Time [min]', size = 15)
    ax3.set_ylabel('Time [min]', size = 15)
    
    # add colorbar
    cb1 = Colorbar(ax = cbax1, mappable = im1)
    cb1.set_label('Flux [arbitrary units]', size = 15)
    
    # save figure 3a
    plt.savefig('../output/figure3/' + str(fignames[n]) + '.png')
    plt.savefig('../output/figure3/' + str(fignames[n]) + '.pdf', dpi = 300)
    plt.show()