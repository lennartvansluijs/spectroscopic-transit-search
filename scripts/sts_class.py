#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Class of Spectral Time Series. This is the main class
    used to store all objects related to the UVES observational data
    of Beta Pictoris in a ordered way.
"""
#%%
import os
import numpy as np
from astropy.io import fits
from scipy.stats import binned_statistic
import scipy.interpolate
from aux import *
from init_func import *
from pulsationcorr_func import *
from matplotlib import pyplot as plt

#%%
class SpectralTimeSeries:
    
    def __init__(self):
        """
        Description:
            Initalize empty Spectral Time Series (STS) class.
        Attributes:
            time - list of timestamps.
            obsid - list of epoch IDs.
            datacube - 2D numpy array with intensities I(t, v)
                       with t time v radial velocity.
            datacube_all_lines - 3D numpy array with intensities I(l, t, v)
                                 with l the line number, t time
                                 and v radial velocity. Full line profiles.
            obsid_dict - dictionary with (key, value) = (obsID, indices). For
                         example obsid_dict[101] = [98,99,100]
                         means datacube[:,98:100] belong to the
                         observation with observation ID '101'.
        """
        
        # initizalize class parameters as None
        self.time = None
        self.obsid = None
        self.radvel = None
        self.obsid_dict = None
        self.datacube = None
        self.datacube_all_lines = None
        
    def init_from_obs(self, inputfolder, outputfolder):
        """
        Description:
            Initalize the class parameters from the observational data.
        Input:
            inputfolder - contains observational data
            outputfolder - save plots and generated data here
        """
        
        # create folder if folder does not exist yet
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        
        # two folders containing time and spectral data
        inputfolder_time = os.path.join(inputfolder, 'frame_times')
        inputfolder_spectral = os.path.join(inputfolder, 'data_individual_lines' \
                                    '_full_profiles')
        inputfolder_flags = os.path.join(inputfolder, 'flags')
        
        self.time = init_time(inputfolder_time)
        self.obsid = init_obsid(inputfolder_time)
        self.radvel = init_radvel(inputfolder_spectral)
        self.obsid_dict = init_obsid_dict(self.obsid)
        self.datacube, self.datacube_all_lines \
        = init_datacube(inputfolder_spectral, inputfolder_flags,
                        self.time, self.radvel,
                        self.obsid_dict, outputfolder)
        
    def load(self, inputfolder, fname):
        """
        Description:
            Load class parameters from saved files.
        Input:
            inputfolder - folder that contains the files
            fname - file name.
        """
        
        # load data from files
        fpath = os.path.join(inputfolder, fname)
        self.datacube = fits.getdata(fpath + '.fits')
        self.time = np.loadtxt(fpath + '_time.txt')
        self.obsid = np.loadtxt(fpath + '_obsid.txt').astype(int).astype(str)
        self.radvel = np.loadtxt(fpath + '_radvel.txt')
        self.obsid_dict = np.load(fpath + '_obsid_dict.npy').item()
            
    def save(self, outputfolder, fname):
        """
        Description:
            Save class parameters to files.
        Input:
            outputfolder - folder where to save class parameters
            fname - file name
        """
        
        # create outputfolder if it does not exist yet
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        
        # save data to files
        fpath = os.path.join(outputfolder, fname)
        wfits(fpath + '.fits', self.datacube)
        np.savetxt(fpath + '_time.txt', self.time)
        np.savetxt(fpath + '_radvel.txt', self.radvel)
        np.savetxt(fpath + '_obsid.txt', self.obsid, fmt="%s")
        np.save(fpath + '_obsid_dict.npy', self.obsid_dict)
    
    def remove_NaN(self):
        """
        Description:
            Remove all slices in the datacube that contain NaN values.
        """
        
        # slices containing NaN values
        mask = np.invert(np.isnan(self.datacube[0, :]))
        
        # remove those slices
        self.datacube = self.datacube[:, mask]
        self.datacube_all_lines = self.datacube_all_lines[:, :, mask]
        self.time = self.time[mask]
        self.obsid = self.obsid[mask]
        self.obsid_dict = init_obsid_dict(self.obsid)
        
    def inject_model(self, rmm, outputfolder, plot = True):
        """
        Description:
            Inject transit model into the original data. Injection is
            performed directly into the combined datacube.
        Input:
            rmm - RMModel class, see rmm_class.py for more information
            outputfolder - outfolder to save plots to
            plot - if True, create plots of the injection routine
        """
        
        # create outputfolder for plots if folder does not exist yet
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        
        # inject planet sogma; omtp tje data
        self.datacube[:, self.obsid_dict[rmm.obsid]] += rmm.model.T

        # create plot of injection routine    
        if plot:
            plot_injection(self.datacube, self.obsid_dict, rmm, outputfolder)
            
    def simnoise(self, spectsnr):
        """
        Description:
            White noise simulation of a Spectral Time Series.
        Input:
            spectsnr - spectral signal to noise ratio
        """

        # simulate observed lineprofiles observed at a noise level sigma
        self.datacube = np.random.normal(0, 1./spectsnr, self.datacube.shape)
        
        # preserve flagged data
        mask = np.isnan(self.datacube[0, :])
        self.datacube[:, mask] = np.nan
        
    def correct_for_pulsations(self, crange = [-2000, 2000, 50], dw = 100):
        """
        Description:
            Apply stellar pulsation correction to the datacube.
        Input:
            crange - list containing shear-constants
                     to use to shear the spectral time series
            dw - integer number of pixels added to the spectral time series to
                 be able to shear the data without information loss at the
                 edges
        """
        
        # create a coppy of the datacube
        datacube_org = np.copy(self.datacube)
        
        # add a border to allow shearing of the datacube
        datacube_s = add_border(self.datacube, dw)
        datacube_s[:] = 0.
        datacube_model = np.copy(datacube_s)
        
        # apply correction per epoch
        for n, (k, v) in enumerate(self.obsid_dict.items()):
            
            # const: shearing constants that determine the amount of shearing
            const = np.linspace(crange[0], crange[1], crange[2])
            medprofs = np.zeros((len(const), datacube_s.shape[0]))
            
            # get spectral time series for this specific observation
            sts_obs = self.datacube[:, v]
            sts_time = self.time[v]
            sts_obs = add_border(sts_obs, dw)
            
            # shear epoch for different amounts of shearing and take mean
            for m, c in enumerate(const):
                
                sts_obs_s = shear(sts_time, sts_obs, c)[0]
                medprofs[m, :] = np.mean(sts_obs_s, axis = 1)
            
            # symmetry correction to correct for possible exoplanet signal
            medprofs[int(medprofs.shape[0]/2.)+1:,:] -= \
            np.flipud(medprofs[0:int(medprofs.shape[0]/2.)-1,:])
            medprofs[:int(medprofs.shape[0]/2.)+1,:] = 0.0
            
            # determine best shearin constant
            ind = np.argmax(np.sum(np.abs(medprofs[:, 75:-75]), axis = 1))

            # create a model of the stellar pulsations
            model_obs = np.zeros(sts_obs.shape)
            model_obs_s, offsets = shear(sts_time, model_obs, const[ind])
            model_obs_s += np.meshgrid(np.arange(model_obs.shape[1]),
                                       medprofs[ind, :])[1]
            model_obs = reverse_shearing(model_obs_s, offsets)
            
            # subtract off the best model
            residual = sts_obs - model_obs
            datacube_s[:, v] = residual
            datacube_model[:, v] = model_obs
        
        # get rid of border again
        datacube_corr = datacube_s[int(dw/2.):-int(dw/2.),:]
        datacube_model = datacube_model[int(dw/2.):-int(dw/2.),:]
        
        # update class parameters            
        self.datacube = np.copy(datacube_corr)
        
#%%
"""
sts = SpectralTimeSeries()
sts.init_from_obs('../data', '../output/sts_init')
sts.remove_NaN()
sts.save('../output/sts_data', 'sts')

inputfolder = '../output/sts_data'
sts = SpectralTimeSeries()
sts.load(inputfolder, 'sts')
#sts.simnoise(spectsnr = 1200)
sts.correct_for_pulsations()

Rp = 1.0
radvel = np.linspace(-100, 100, 146)
A = 0.8
b = 0.
theta= 0.
x0 = np.linspace(-1.3, 1.3, 3)
rmm = RMModel(Rp, radvel, A, b, theta, x0)
rmm.simtransit('../output/rm_model')
x0_ind = 1
rmm.obsid = '58004'
rmm.simple_static_model(sts, x0_ind)

sts.inject_model(rmm, '../output/rm_model', plot = True)
"""