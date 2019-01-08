#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Class to perform a planet injection routine.
"""
#%%
import os
import numpy as np
from astropy.io import fits
from sts_class import SpectralTimeSeries
from rmm_class import RMModel
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import matplotlib.gridspec as gridspec
from aux import *
from pir_func import *
    
#%%

class PlanetInjectionRoutine:
    """
    Description:
        This class contains all the relevant functions and parameters
        of injection multiple planet signals and running statistics on the
        obtained results.
    Attributes:
        Rp - list of planet radii to use for the injections [Rjup]
        b - list of impact parameters to use for the simulations
        A - float intrinsic line depth used for planet models
        outputfolder - main outputfolder to save data to
        sts - use this STS class object for the data injection
        x0 - list of positions where to simulate transits [Rstar]
        spectsnr - in case of a white noise simulation, use this spectral SNR
        mode - mode used for the injection. 'wn': white noise simulation or
               'spcorr': stellar pulsation correction
    """
    
    def __init__(self, Rp, b, A, theta, x0, outputfolder,
                 sts, mode, spectsnr = None):
        
        # create outputfolder if it does not exist yet
        if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)
                
        # initalize class parameters
        self.outputfolder = outputfolder
        self.Rp = Rp
        self.b = b
        self.A = A
        self.theta = theta
        self.sts = sts
        self.x0 = x0
        self.mode = mode
        self.spectsnr = spectsnr
        
    def runsim(self):
        """
        Description:
            Run all simulations to create the RM-planet models used for
            the injection.
        """
        
        # initizalize model info and create subdirectoty to save
        # simulation information
        n = 0
        modelinfo = []
        outputfolder = os.path.join(self.outputfolder, 'runsim')
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        
        # create txt file for model information
        if os.path.exists(os.path.join(outputfolder, 'modelinfo.txt')):
            os.remove(os.path.join(outputfolder, 'modelinfo.txt'))
        createtxt(os.path.join(outputfolder, 'modelinfo.txt'))
        appendheader(os.path.join(outputfolder, 'modelinfo.txt'),
                     '# column 1: model id, column 2: planet radius Rp [Rjup]'\
                     ', column 3: impact parameter b')
        
        # simulate for different planet radii and impact parameters
        for i in range(len(self.Rp)):
            for j in range(len(self.b)):
                
                # progress
                n += 1
                print('Simulation ('+str(n)+'/'+ str(len(self.Rp)*\
                                         len(self.b))+')')
                
                # create subdirectory for this simulation
                outputfolder_sim = os.path.join(outputfolder,
                                                'model_' + str(n))
                if not os.path.exists(outputfolder_sim):
                    os.makedirs(outputfolder_sim)
                
                # create model
                rmm = RMModel(self.Rp[i], self.sts.radvel, self.A, self.b[j],
                              self.theta, self.x0)
                rmm.simtransit(outputfolder_sim)
                appendlist(os.path.join(outputfolder, 'modelinfo.txt'),
                           ['model_' + str(n), str(self.Rp[i]), str(self.b[j])])
                
    def runinj(self, plot = False):
        """
        Description:
            Inject planet in all observations and use
            median subtraction as data reduction method.
        Input:
            plot - slows down code, but visualizes injected datacubes
        """
        
        # original and corrected datacube determined for desired data
        # reduction mode
        print('Used mode: '+str(self.mode))
        if self.mode == 'spcorr':
            datacube_org = np.copy(self.sts.datacube)
        elif self.mode == 'wn':
            self.sts.simnoise(spectsnr = self.spectsnr)
            datacube_org = np.copy(self.sts.datacube)
        
        # create txt file
        outputfolder = os.path.join(self.outputfolder, 'runinj')
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        if os.path.exists(os.path.join(outputfolder, 'injinfo.txt')):
            os.remove(os.path.join(outputfolder, 'injinfo.txt'))
        createtxt(os.path.join(outputfolder, 'injinfo.txt'))
        appendheader(os.path.join(outputfolder, 'injinfo.txt'),
                     '# column 1: injection id, column 2: planet radius, '\
                     'column 3: impact parameter b, ' \
                     'column 4: position [Rs]')
        
        # load info of simulated models
        modelinfo = np.loadtxt(os.path.join(self.outputfolder,
                                            'runsim/modelinfo.txt'),
        dtype = 'str')
        
        # loop over all models at all injected radial velocities
        n = 0
        for i in range(modelinfo.shape[0]):
            for j in range(len(self.x0)):
                
                # progress
                n += 1
                print('Injection ('+str(n)+'/'+str(modelinfo.shape[0]*\
                                  len(self.x0))+')')
                
                # create subdirectory for injection data
                outputfolder_inj = os.path.join(outputfolder, 'inj_' + str(n))
                if not os.path.exists(outputfolder_inj):
                    os.makedirs(outputfolder_inj)
                    
                # load model parameters
                Rp = modelinfo[i, 1]
                b = modelinfo[i, 2]
                pos = self.x0[j]
                
                # update datacube
                self.sts.datacube = np.copy(datacube_org)
                
                # create injection model
                rmm = RMModel(Rp, self.sts.radvel, None, None, None, self.x0)
                rmm.fullmodel = fits.getdata(self.outputfolder + \
                                             '/runsim/model_' + \
                                             str(i+1) + \
                                             '/diff_lineprof.fits')
                
                # inject model in all epochs
                for obsid in self.sts.obsid_dict.keys():
                    rmm.obsid = str(obsid)
                    rmm.simple_static_model(self.sts, x0_ind = j)
                    self.sts.inject_model(rmm, outputfolder_inj, plot = False)
                
                # assign memory for injection datacube
                datacube_inj = np.copy(self.sts.datacube)
                  
                # apply desired mode of data reduction
                if self.mode == 'spcorr':
                    self.sts.correct_for_pulsations()
                elif self.mode == 'wn':
                    pass
                
                # this is the datacube that has been injected and corrected
                datacube_inj_corr = np.copy(self.sts.datacube)
                
                # plot datacubes if desired
                if plot:
                    plot_result(datacube_org,
                                datacube_inj,
                                datacube_inj_corr, outputfolder_inj)
                    
                # save the final datacube for later analysis
                np.save(os.path.join(outputfolder_inj, 'datacube.npy'),
                        self.sts.datacube)
                np.save(os.path.join(outputfolder_inj, 'model.npy'),
                        rmm.model)
                appendlist(os.path.join(outputfolder, 'injinfo.txt'),
                           ['inj_' + str(n), str(Rp), str(b), str(pos)])
                
        # set back to default value
        self.sts.datacube = np.copy(datacube_org)
    
    def getstats(self):
        """
        Description:
            Obtain SNR statistics from the injections.
        """
        
        # set back to default value
        print('Used mode: '+str(self.mode))
        if self.mode == 'spcorr':
            datacube_org = np.copy(self.sts.datacube)
            self.sts.correct_for_pulsations()
            datacube_corr = np.copy(self.sts.datacube)
        elif self.mode == 'wn':
            self.sts.simnoise(self.spectsnr)
            datacube_org = np.copy(self.sts.datacube)
            datacube_corr = np.copy(self.sts.datacube)
        
        # setup .txt file containing statistics for every injection
        injinfo = np.loadtxt(os.path.join(self.outputfolder,
                                          'runinj/injinfo.txt'), dtype = 'str')
        outputfolder = os.path.join(self.outputfolder, 'getstats')
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        stats = np.zeros((len(self.Rp), len(self.x0),
                          len(self.b), len(self.sts.obsid_dict.keys()), 5))
        
        # loop over all injected signals
        for n in range(injinfo.shape[0]):
            
            # progress
            print('Get statistics ('+str(n+1)+'/'+str(injinfo.shape[0])+')')

            # get datacube from generated injection data
            datacube = np.load(os.path.join(self.outputfolder,
                                 'runinj/inj_' + str(n+1) + '/datacube.npy'))
            model = np.load(os.path.join(self.outputfolder,
                                 'runinj/inj_' + str(n+1) + '/model.npy'))
            
            # obtain the indices of the parameters used for the injection
            ind_Rp = np.argmin( np.abs(self.Rp - float(injinfo[n, 1])) )
            ind_b = np.argmin( np.abs(self.b - float(injinfo[n, 2])) )
            ind_pos = np.argmin( np.abs(self.x0 - float(injinfo[n, 3])) )
               
            # determine the noise estimate of the stellar pulsations from
            # the corrected datacube for this specific model
            noise_sp = getn_sp(datacube_corr, model,
                               self.sts.obsid_dict, binsize = 8)

            for i, (k, v) in enumerate(self.sts.obsid_dict.items()):
                
                # snr for the observation
                signal_obs, noise_obs, snr_obs = \
                getsnr_obs(datacube[:, v], model, dw = 25, binsize = 8)
                
                # snr for the signal/noise estimate of the stellar pulsations
                snr_sp = signal_obs/noise_sp
                
                # add to the datacube used to organize all statistics
                stats[ind_Rp, ind_pos, ind_b, i, 0] = signal_obs
                stats[ind_Rp, ind_pos, ind_b, i, 1] = noise_obs
                stats[ind_Rp, ind_pos, ind_b, i, 2] = snr_obs
                stats[ind_Rp, ind_pos, ind_b, i, 3] = noise_sp
                stats[ind_Rp, ind_pos, ind_b, i, 4] = snr_sp
                
        # save the datacube containing all stats
        wfits(os.path.join(outputfolder, 'stats.fits'), stats)
        
    def plot_sensitivity(self, sigma, veq):
        """
        Description:
            Create a final plot of the sensitivity.
        Input:
            sigma - SNR limit adopted
            veq - v sin i value
        """
        
        # create folder to store the results
        outputfolder = os.path.join(self.outputfolder, 'results')
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
            
        # obtain generated statistics
        statscube = fits.getdata(os.path.join(self.outputfolder,
                                              'getstats/stats.fits'))
        
        # determine fraction above sigma level
        fractioncube = np.zeros(statscube[:, :, 0, :, 0].shape)
        mask = (statscube[:, :, 0, :, 4] > sigma)
        fractioncube[mask] = 1.
        fraction = np.sum(fractioncube[:, 1:-1], axis = 2) \
        / fractioncube.shape[2] * 100 # perc
    
        # plot data
        fig, ax = plt.subplots()
        cax = plt.imshow(np.flipud(fraction), extent = [0, 1, 0, 1],
                        aspect = 1.,
                        cmap = discrete_cmap(10, base_cmap='viridis'),
                        vmin = 0, vmax = 100)
    
        # set labels of the x and y axes
        xticks = np.arange(len(self.x0[1:-1]))/float(len(self.x0[1:-1]))+1./(2.*float(len(self.x0[1:-1])))
        xticklabels = [str(np.round(x, 1))[:-2] for x in \
                       np.linspace(self.x0[1:-1].min() * veq, self.x0[1:-1].max() * veq, len(self.x0[1:-1]))]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        yticks = np.arange(len(self.Rp))/float(len(self.Rp))+1./(2.*float(len(self.Rp)))
        yticklabels = np.round(np.linspace(self.Rp.min(), self.Rp.max(), len(self.Rp)), 2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(r'Radial velocity [km/s]', size = 15)
        ax.set_ylabel(r'Planet radius [R$_{\rm{Jup}}$]', size = 15)
        plt.tick_params(top=False,
                       bottom=False,
                       left=False,
                       right=False,
                       labelleft=True,
                       labelbottom=True)

        # plot position of planets in the solar system
        Rjup = 1.0
        Rsat = 0.832
        Rnep = 0.352
        lcolor = 'k'
        labelcolor = 'w'
        ndots = 25
        plabels = ['Jupiter', 'Saturn', 'Neptune']
        if self.mode == 'wn':
            pcolor = ['k', 'k', 'w']
        elif self.mode == 'spcorr':
            pcolor = ['w', 'w', 'w']
        for n, ypos in enumerate([Rjup, Rsat, Rnep]):
            plt.scatter(np.linspace(0, 1, ndots), np.zeros(ndots) + ypos-0.05,
                       color = pcolor[n], s = 7.50)
            plt.text(x = 0.02, y = ypos - 0.08, s = plabels[n], ha = 'left', va = 'center',
                    size = 11, color = pcolor[n])
        plt.xlim(0,1)
        
        # set limits of the colorbar
        cbar = fig.colorbar(cax, ticks=[0, 20, 40, 60, 80, 100])
        cbar.ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        cbar.set_label('Fraction recovered', rotation=270, labelpad = 10, size = 15)


        # make plot look a bit better
        #plt.tight_layout()
        
        # save as a .png and .pdf file
        plt.savefig(os.path.join(outputfolder, 'recfrac.png'),
                    dpi = 300)
        plt.savefig(os.path.join(outputfolder, 'recfrac.pdf'))
        
#%%
"""
sts = SpectralTimeSeries() 
sts.load('../output/sts_data', 'sts')

Rp = np.linspace(0.1, 1.0, 10)
b = np.linspace(0, 1.0, 1)
A = 0.8
theta = 0
mode = 'spcorr'
outputfolder = '../output/pir_spcorr'
x0 = np.array([-200., -120., -100., -80., -60., -40., -20., 0., 20.,
          40., 60., 80., 100., 120., 200.])/130.
        
pir = PlanetInjectionRoutine(Rp, b, A, theta, x0, outputfolder,
                             sts, mode, spectsnr = None)

pir.runsim()
pir.runinj(plot = False)
pir.getstats()
pir.plot_sensitivity(sigma = 3.0, veq = 130)
"""