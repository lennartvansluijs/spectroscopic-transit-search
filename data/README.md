# Data

This repository contains the following files:
* **frame_times** contains the timestamps for every observations, used to calculate the observation window function
* **rm_models** contains the RM-models used for to model the two candidate signals
* **stellarparams.txt** contains the stellar parameters as a function of spectral type
* **sts.fits** spectral time series after subtracting the median line profiles and combining all 16 lines
* **sts_all_lines.fits** spectral time series after subtracting the median line profiles of all 16 lines
* **sts_obsid.txt** list of observation IDs
* **sts_obsid_dict.npy** dictionary of observation IDs
* **sts_radvel** list of radial velocities on first dimension of sts.fits [km/s]
* **sts_time** list of timestamps on second dimension of sts.fits [JD]


