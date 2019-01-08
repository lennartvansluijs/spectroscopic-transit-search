""" 
Code to generate line profiles for the RM effect of rings taking into account the pixel size of the data and the resolution of the spectrograph.

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
A=0.8                                                      # line depth
veq=130.                                                   # V sin i (km/s)
l_fwhm=20.                                                 # Intrinsic line FWHM (km/s)
lw=l_fwhm/2.35                                             # Intinsice line width in sigma (km/s)

vgrid=np.arange(-250,250.1,0.25)                           # velocity grid
profile=make_lineprofile(npix_star,Rs,xc,vgrid,A,veq,lw)   # make line profile for each vertical slice of the star

star=make_star(npix_star,OSF,xc,yc,Rs,u1,u2,map0)          # make a limb-darkened stellar disk
sflux=star.sum(axis=1)                                     # sum up the stellar disk across the x-axis
improf=np.sum(sflux[:,np.newaxis]*profile,axis=0)          # calculate the spectrum for an unocculted star

normalisation=improf[0]
improf=improf/normalisation

#set up the orbit including a spin-orbit misalignment
phase=(np.arange(101)/50.-1.)*0.02
a_Rs=10.
b=0.3
theta=25.*np.pi/180.  #projected spin-orbit misalignment

x0=a_Rs*np.sin(2.*np.pi*phase)
y0=b*np.cos(2.*np.pi*phase)
x1=(x0*np.cos(theta)-y0*np.sin(theta))*Rs+xc
y1=(x0*np.sin(theta)+y0*np.cos(theta))*Rs+yc


#define some basic arrays
line_profile1=np.zeros( (x1.shape[0],vgrid.shape[0]) )
img1=np.zeros( (x1.shape[0],npix_star,npix_star) )

#Loop over the positions of the planet
for i in range(x1.shape[0]):    
    tmap1=star* make_planet(npix_star,OSF,y1[i],x1[i],Rp,map0)
    sflux=tmap1.sum(axis=0)
    line_profile1[i,:]=np.dot(sflux,profile)/normalisation
    img1[i,:,:]=tmap1

#Calculate the differential profile as used for beta Pic
diff_lineprof=line_profile1/((line_profile1[0,:])[np.newaxis,:])-1.
diff_lineprof=diff_lineprof-(diff_lineprof[:,0])[:,np.newaxis]

#write the arrays to disk
hdu=fits.PrimaryHDU(img1)
hdu.writeto('ctmp.fits',overwrite=True)
hdu=fits.PrimaryHDU(line_profile1)
hdu.writeto('ctmp2.fits',overwrite=True)
hdu=fits.PrimaryHDU(diff_lineprof)
hdu.writeto('ctmp3.fits',overwrite=True)
