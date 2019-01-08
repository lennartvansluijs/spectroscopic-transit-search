#!/usr/bin/env python3
#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Author:
    Lennart van Sluijs
Date:
    Dec 21 2018
Description:
    Functions used to initalize the STS class from the UVES data.
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib import gridspec
from aux import *
from astropy.io import fits
from matplotlib import cm as cm
from matplotlib import figure
import warnings

#%%
def init_time(inputfolder):
    """
    Description:
        Create a list of times for all time frames of all observations.
    Input:
        inputfolder - contains time data for all observations
    Output:
        time - list of times of all observations
    """
    
    # list all files
    flist = get_flist(inputfolder)
    
    # create list of times
    time = []
    for fname in flist:
        fname_time = np.loadtxt(os.path.join(inputfolder, fname))
        time = np.append(time, fname_time)

    # convert to numpy array for later usage
    time = np.array(time)
    
    return time

def init_obsid(inputfolder):
    """
    Description:
        Create a list of observation ID's for all time frames
        of all observations.
    Input:
        inputfolder - contains time data for all observations
    Output:
        obsid - list of times of all observations
    """
    
    # list all files
    flist = get_flist(inputfolder)
    
    # create list of times
    obsid = []
    for fname in flist:
        fname_time = np.loadtxt(os.path.join(inputfolder, fname))
        obsid = np.append(obsid,
                          [fname[0:5] for n in range(len(fname_time))])

    # convert to numpy array for later usage
    obsid = np.array(obsid)
    
    return obsid

def init_radvel(inputfolder):
    """
    Description:
        Create a list of radial velocities for all time frames
        of all observations.
    Input:
        inputfolder - contains time data for all observations
    Output:
        radvel - list radial velocities
    """
    
    # list all files that contain radial velocity data
    flist = findall_containing_string(get_flist(inputfolder), 'vel',
                                      inelement = True)
    
    # pick first file as radvel is similar for all observations
    radvel = fits.getdata(os.path.join(inputfolder, flist[0]))[:]
    
    # convert to numpy array for later usage
    radvel = np.array(radvel)

    return radvel

def init_obsid_dict(obsid):
    """
    Description:
        Initalize dictionary of observation ID's.
    Input:
        obsid - list of all observation ID's for all time frames
                for all observations
    Output:
        obsid_dict - dictionary of observation ID's
    """
    
    # get number of obsid's
    unique, counts = np.unique(obsid, return_counts=True)
    
    # create dictionary with (key, value) = (obsid, indices in datacube)
    obsid_dict = {}
    n = 0
    for index, obs in enumerate(unique):
        obsid_dict[obs] = np.arange(counts[index]) + n
        n += counts[index]
    
    return obsid_dict

def flag_flist(flist, line, flags_fpath,
               flag_noline = True, flag_blends = True,
               flag_noisy = False):
    """
    Description:
        Flag observations in file list if observation in flags.txt.
    Input:
        flist - list of files
        flags_fpath - path of the flag.txt file
        flag_noline - flag files where line is not visible
        flag_blends - flag files where line blends with background
        flag_noisy - flag files which are extra noisy
    Output:
        flist_flagged - list of files with flagged observations removed
    """
    
    # get file containing flag information
    flags = np.loadtxt(flags_fpath, dtype= 'str')[line]

    # flag data
    remove_ind = []
    for index, fname in enumerate(flist):
        if flag_noline:
            if fname[23:28] in flags[1].split(','):
                remove_ind.append(index)
        if flag_blends:
            if fname[23:28] in flags[2].split(','):
                remove_ind.append(index)
        if flag_noisy:
            if fname[23:28] in flags[3].split(','):
                remove_ind.append(index)
                
    # list all file names that should be flagged
    flist_flagged = [fname for index, fname in enumerate(flist) \
                     if index not in remove_ind]

    return flist_flagged
    
def plot_lineprofs(datacube, nlines, outputfolder, fname):
    """
    Description:
        Plot lineprofiles of all lines.
    Input:
        datacube - 3D numpy array with dimensions
                   (# line, radvel, # time frame)
        nlines - total number of lines in the datacube
        outpufolder - save plot here
        fname - name of the plot
    """
    
    # initalize figure
    fig, axes = plt.subplots(nlines, 1)
    #spec = fig.add_gridspec(nlines, 1)
    plt.suptitle('All observations of all lines', fontsize = 12, y = 0.915)
    
    # plot spectral time series for all epochs of this line profile
    for line, row in enumerate(axes):
        
        cmap = cm.Greys_r
        cmap.set_bad('dimgrey',1.)
        axes[line].imshow(datacube[line,:,:], cmap)
        axes[line].get_xaxis().set_visible(False)
        axes[line].get_yaxis().set_visible(False)
        axes[line].annotate(str(line+1), xy = (-0.025,0.5), xycoords='axes fraction',
                    va='center', ha='center', fontsize = 10)
    
    # add labels
    fig.add_subplot(axes[15]).annotate('# time frame',
                   xy = (0.5,-0.6), xycoords='axes fraction',
                   va='center', ha='center', fontsize = 12)
    fig.add_subplot(axes[0]).annotate('# line',
                   xy = (-0.06,-8), xycoords='axes fraction', rotation = 90,
                   va='center', ha='center', fontsize = 12)
    fig.add_subplot(axes[0]).annotate('radial velocity',
                   xy = (1.04,-8), xycoords='axes fraction', rotation = -90,
                   va='center', ha='center', fontsize = 12)
    fig.add_subplot(axes[0]).annotate('}',
                   xy = (1.014,-7.9), xycoords='axes fraction',
                   va='center', ha='center', fontsize = 14)
    
    # save plot and close
    plt.savefig(os.path.join(outputfolder, fname + '.pdf'))
    plt.savefig(os.path.join(outputfolder, fname + '.png'), dpi = 300)
    plt.close()

def init_datacubes_lineprof(inputfolder, inputfolder_flags,
                            time, radvel, obsid_dict,
                            outputfolder = '', nlines = 16,
                            save = True, plot = True):
    """
    Description:
        Initalize datacubes with lineprofiles from the input data.
    Input:
        inputfolder - contains input data
        time - list of times for all time frames of all observations
        radvel - list of rafial velocities for all time frames
                 of all observations
        obsid_dict - dictionary of observation ID's
        outputfolder - save plots and datacubes here
                       if plot and/or save is True
        nlines - total number of lines
        save - save datacubes if True
        plot - create plots if True
    """
    
    # list all apectral time series data (no 'vel' in file name)
    flist = findall_containing_string(get_flist(inputfolder), 'vel',
                                      inelement = False)

    # initialize 3 datacubes for all lines: one containing the line profiles,
    # one containing the reference line profiles and one where everything
    # has been subtracted
    datacube_lineprofs = np.zeros((nlines, len(radvel), len(time)))
    datacube_ref_lineprofs = np.zeros((nlines, len(radvel), len(time)))
    datacube_res_lineprofs = np.zeros((nlines, len(radvel), len(time)))
    datacube_lineprofs[:] = np.nan
    datacube_ref_lineprofs[:] = np.nan
    datacube_res_lineprofs[:] = np.nan
    
    # get reference and residual line profiles
    flags_fpath = os.path.join(inputfolder_flags, 'flags.txt')
    for line in np.arange(nlines):
        
        flist_flagged = flag_flist(flist, line, flags_fpath = flags_fpath)
        for fname in flist_flagged:
            
            # open lineprofile data from file
            fpath = os.path.join(inputfolder, fname)
            obsid = fname[23:28]
            lineprof = fits.getdata(fpath)[line][:][:]
            datacube_lineprofs[line, :, obsid_dict[obsid]] = lineprof

        # reference line profile is renormalized median of all nights
        ref_lineprof = np.nanmedian(datacube_lineprofs[line][:][:], axis = 1)
        normalisation = np.mean(ref_lineprof[0:20])
        datacube_lineprofs[line, :, :] /= normalisation
        ref_lineprof /= normalisation
        ref_lineprof = np.meshgrid(np.arange(len(time)), ref_lineprof)[1]
        res_lineprof = datacube_lineprofs[line][:][:] - ref_lineprof
        
        # add to datacubes
        datacube_ref_lineprofs[line, :, :] = ref_lineprof
        datacube_res_lineprofs[line, :, :] = res_lineprof

    # create plots of the 3 datacubes
    if plot:
        plot_lineprofs(datacube_lineprofs, nlines,
                       outputfolder, 'lineprofs')
        plot_lineprofs(datacube_ref_lineprofs, nlines,
                       outputfolder, 'ref_lineprofs')
        plot_lineprofs(datacube_res_lineprofs, nlines,
                       outputfolder, 'res_lineprofs')
    
    # save 3 datacubes as .fits files
    if save:
        wfits(os.path.join(outputfolder, 'datacube_lineprofs.fits'),
              datacube_lineprofs)
        wfits(os.path.join(outputfolder, 'datacube_ref_lineprofs.fits'),
              datacube_ref_lineprofs)
        wfits(os.path.join(outputfolder, 'datacube_res_lineprofs.fits'),
              datacube_res_lineprofs)
    
    return datacube_lineprofs, datacube_ref_lineprofs, datacube_res_lineprofs

def plot_datacube(datacube, radvel, outputfolder, fname):
    """
    Description:
        Plot spectral time series in datacube.
    Input:
        datacube - 2D numpy array with dimensions (radvel, # time frame)
        radvel - list of radial velocities
        outpufolder - save plot here
        fname - name of the plot
    """
    
    # initalize figure
    fig = plt.figure()
    
    # use grey scale
    cmap = cm.Greys_r
    cmap.set_bad('dimgrey', 1.)
    
    # plot data
    plt.imshow(datacube, cmap,
               aspect=4, extent=[0, datacube.shape[1],
                                 np.min(radvel), np.max(radvel)])
    plt.xlabel('# time frame', size = 12)
    plt.ylabel('Radial velocity [km/s]', size = 12)
    plt.title('All observations of all lines combined')

    # save plot and close
    plt.savefig(os.path.join(outputfolder, fname + '.pdf'))
    plt.savefig(os.path.join(outputfolder, fname + '.png'), dpi = 300)
    plt.close()

def init_datacube(inputfolder, inputfolder_flags,
                  time, radvel, obsid_dict,
                  outputfolder = '', nlines = 16,
                  save = True, plot = True):
    """
    Description:
        Initalize datacube with residual line profile.
    Input:
        inputfolder - contains input data
        inputfolder_flags - contains flags.txt input data
        time - list of times for all time frames of all observations
        radvel - list of rafial velocities for all time frames
                 of all observations
        obsid_dict - dictionary of observation ID's
        outputfolder - save plots and datacubes here
                       if plot and/or save is True
        nlines - total number of lines
        save - save datacubes if True
        plot - create plots if True
    Output:
        datacube_res - 2D numpy array the combined STS of all lines
        datacube_lineprofs - 3D numpy array with the STS of all full lines
    """
    
    # first initalize the datacube for all of the lines
    datacube_lineprofs, datacube_ref_lineprofs, datacube_res_lineprofs = \
    init_datacubes_lineprof(inputfolder, inputfolder_flags, time, radvel,
                            obsid_dict, outputfolder, nlines, save, plot)
    
    # secondly, combine all of the lines. Ignore runtime warnings in this
    # block for cases where all nights were flagged
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        datacube = np.nanmean(datacube_lineprofs, axis = 0)
        datacube_ref = np.nanmean(datacube_ref_lineprofs, axis = 0)
        datacube_res = np.nanmean(datacube_res_lineprofs, axis = 0)

    # plot all datacubes
    if plot:
        plot_datacube(datacube, radvel, outputfolder, 'datacube')
        plot_datacube(datacube_ref, radvel, outputfolder, 'datacube_ref')
        plot_datacube(datacube_res, radvel, outputfolder, 'datacube_res')
        
    # save all datacubes
    if save:
        wfits(os.path.join(outputfolder, 'datacube.fits'), datacube)
        wfits(os.path.join(outputfolder, 'datacube_ref.fits'), datacube_ref)
        wfits(os.path.join(outputfolder, 'datacube_res.fits'), datacube_res)
    
    return datacube_res, datacube_lineprofs