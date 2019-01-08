#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Run this script to generate Figure 5.
"""

import numpy as np
from pir_class import *
from sts_class import *

# initialize from lineprofile cuts of the observational data
# by the UVES spectograph.
sts = SpectralTimeSeries()
sts.init_from_obs('../data', '../output/figure5/sts_init')
sts.remove_NaN()
sts.save('../output/figure5/sts_data', 'sts')

# reload spectral time series
sts = SpectralTimeSeries()
sts.load('../output/figure5/sts_data', 'sts')

# parameters for the planet injection routine
Rp = np.linspace(0.1, 1.0, 10) # planet radii injected [Rjup]
b = np.linspace(0, 1.0, 1) # impact parameter injected
A = 0.8 # intrinsic line depth
theta = 0 # spin-orbit misalignment
mode = 'spcorr' # planet injection routine mode
outputfolder = '../output/figure5/pir_spcorr' # outputfolder
veq = 130 # v sin i Beta Pic [km/s]
x0 = np.array([-200., -120., -100., -80., -60., -40., -20., 0., 20.,
          40., 60., 80., 100., 120., 200.])/veq # positions in front of star
snrlim = 3.0 # limit for the sensitivity adopted

# run planet injection routine
pir = PlanetInjectionRoutine(Rp, b, A, theta, x0, outputfolder,
                             sts, mode, spectsnr = None)
pir.runsim() # simulate lineprofiles
pir.runinj(plot = False) # inject signals
pir.getstats() # get snr
pir.plot_sensitivity(sigma = snrlim, veq = veq) # create plot

# save and show Figure 1
plt.savefig('../output/figure5/figure5.png', dpi = 300)
plt.savefig('../output/figure5/figure5.pdf')
plt.show()