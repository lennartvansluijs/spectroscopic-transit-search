#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Run this script to generate the data used to determine the sensitivity
    limits by running a planet injection routine.
"""

#%%
import numpy as np
from pir_class import *
from sts_class import *

#%%
"""
Create a Spectral Time Series from all observational data files
by also flagging certain epochs.
"""
"""
sts = SpectralTimeSeries()
sts.init_from_obs('../data', '../output/sts_init')
sts.remove_NaN()
sts.save('../output/sts_data', 'sts')
"""
#%%
"""
Run the planet injection routine for a white noise simulation.
"""
"""
# reload spectral time series
sts = SpectralTimeSeries() 
sts.load('../output/sts_data', 'sts')

# parameters for the planet injection routine
Rp = np.linspace(0.1, 1.0, 10) # planet radii injected [Rjup]
b = np.linspace(0, 1.0, 1) # impact parameter injected
A = 0.8 # intrinsic line depth
theta = 0 # spin-orbit misalignment
mode = 'wn' # planet injection routine mode
spectsnr = 1200 # spectral SNR used for the white noise simulation
outputfolder = '../output/pir_wn' # outputfolder
veq = 130 # v sin i Beta Pic [km/s]
x0 = np.array([-200., -120., -100., -80., -60., -40., -20., 0., 20.,
          40., 60., 80., 100., 120., 200.])/veq # positions in front of star
snrlim = 3.0 # limit for the sensitivity adopted
        
pir = PlanetInjectionRoutine(Rp, b, A, theta, x0, outputfolder,
                             sts, mode, spectsnr)
#pir.runsim()
pir.runinj(plot = False)
pir.getstats()
pir.plot_sensitivity(sigma = snrlim, veq = veq)
"""

#%%
"""
Run the planet injection routine for the real data with stellar pulsation
correction.
"""

# reload spectral time series
sts = SpectralTimeSeries() 
sts.load('../output/sts_data', 'sts')

# parameters for the planet injection routine
Rp = np.linspace(0.1, 1.0, 2) # planet radii injected [Rjup]
b = np.linspace(0, 1.0, 1) # impact parameter injected
A = 0.8 # intrinsic line depth
theta = 0 # spin-orbit misalignment
mode = 'spcorr' # planet injection routine mode
outputfolder = '../output/pir_spcorr' # outputfolder
veq = 130 # v sin i Beta Pic [km/s]
x0 = np.array([-200., -120., -100., -80., -60., -40., -20., 0., 20.,
          40., 60., 80., 100., 120., 200.])/veq # positions in front of star
snrlim = 3.0 # limit for the sensitivity adopted

pir = PlanetInjectionRoutine(Rp, b, A, theta, x0, outputfolder,
                             sts, mode, spectsnr = None)
pir.runsim()
pir.runinj(plot = False)
pir.getstats()
pir.plot_sensitivity(sigma = snrlim, veq = veq)