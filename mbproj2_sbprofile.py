#!/usr/bin/env python

"""
Generate a surface brightness profile for feeding into autorebin.py

NaN values are ignored, so use them for masking point sources
"""

# from __future__ import print_function, division

import argparse
from astropy.io import fits
import numpy as N
import warnings

warnings.filterwarnings("ignore", category=N.VisibleDeprecationWarning)

def makeProfile(img, xc, yc, rmax, binf):
    """Take image and compute profile. Ignores nan values.

    Returns: array of total values in bins, array of number of pixels in bins

    xc, yc: centre in pixels
    rmax: maximum radius in pixels
    binf: binning factor
    """

    radii = N.fromfunction(
        lambda y, x: (N.sqrt((x-xc)**2 + (y-yc)**2)/binf).astype(N.int32),
        img.shape)

    rmaxdiv = int(rmax / binf)
    radii[ N.logical_not(N.isfinite(img)) | (radii >= rmaxdiv) ] = rmaxdiv

    numpix = N.bincount(radii.ravel())
    ctsums = N.bincount(radii.ravel(), weights=img.ravel())

    return ctsums[:rmaxdiv], numpix[:rmaxdiv]

def oversampleCounts(inimage, oversample):
    """Oversample by randomizing counts over pixels which are
    oversample times larger."""

    if oversample == 1:
        return inimage

    if inimage.dtype.kind not in 'iu':
        raise ValueError("Non-integer input image")
    if N.any(inimage < 0):
        raise ValueError("Input counts image has negative pixels")

    y, x = N.indices(inimage.shape)

    # make coordinates for each count
    yc = N.repeat(N.ravel(y), N.ravel(inimage))
    xc = N.repeat(N.ravel(x), N.ravel(inimage))

    # add on random amount
    xc = xc*oversample + N.random.randint(oversample, size=len(xc))
    yc = yc*oversample + N.random.randint(oversample, size=len(yc))

    outimages = N.histogram2d(
        yc, xc, (inimage.shape[0]*oversample, inimage.shape[1]*oversample))
    outimage = N.array(outimages[0], dtype=N.int)

    return outimage

def oversampleSimple(inimage, oversample, average=False):
    """Oversample by repeating elements by oversample times."""
    os1 = N.repeat(inimage, oversample, axis=1)
    os0 = N.repeat(os1, oversample, axis=0)

    if average:
        os0 *= (1./oversample**2)

    return os0

def mbextract(inimage, outprofile,rmax, xc, yc, binf, exposuremap, mask):
    """extract a surface brightness profile
    'input image','outprofile','mask=mask_file'
    'xc: x centre (pixels)','yc: y centre (pixels)'
    'rmax: maximum radius (pixels)','exposuremap: exposure map for scaling areas (optional)'
    'binf: bin factor (pixels)','oversample:oversample image by N pixels'
    args: exposuremap, mask
    """
    oversample = 1
    with fits.open(inimage) as f:
        exposure = f[0].header['EXPOSURE']
        pixsize_arcmin = abs(f[0].header['CDELT1'] * 60)
        img = N.array(f[0].data)
        if oversample != 1:
            if img.dtype.kind in 'ui':
                print('Oversampling in integer count mode')
                img = oversampleCounts(img, oversample)
            else:
                print('Oversampling in simple (non-count) mode')
                img = oversampleSimple(img, oversample, average=True)

            # output radii are scaled after resampling
            pixsize_arcmin /= oversample
            binf *= oversample
            rmax *= oversample
            xc *= oversample
            yc *= oversample

    if mask != None:
        with fits.open(mask) as f:
           img = img.astype(N.float64)
           img = oversampleSimple(img, oversample)
           img[f[0].data==0] = N.nan

    if exposuremap:
        with fits.open(exposuremap) as f:
            expmap = N.array(f[0].data)

        expmap = oversampleSimple(expmap, oversample, average=True)
        # make sure same bad pixels used (and use nan in expmap for bad pixels)
        expmap = N.where(expmap != 0, expmap, N.nan)
        expmap = N.where(N.isfinite(img), expmap, N.nan)
        img = N.where(N.isfinite(expmap), img, N.nan)
    else:
        expmap = None

    ctsum, pixsum = makeProfile(img, xc, yc, rmax, binf)
    areas = pixsum * pixsize_arcmin**2
    exposures = N.full_like(ctsum, exposure)

    # if there is an exposure map, scale the exposures by the
    # variation in exposure from the centre to the annulus
    if expmap is not None:
        expsum, exppixsum = makeProfile(expmap, xc, yc, rmax, binf)
        avexp = expsum / exppixsum
        # get estimate of central value in profile
        npix = sumexp = i = 0
        try:
            while npix < 20:
                npix += exppixsum[i]
                sumexp += expsum[i]
                i += 1
            # scale by central value (where rmf is calculated)
            scaledexp = avexp / (sumexp / npix)
        except:
            print('Wrong center')
            scaledexp = 1
            pass
        # scale outputted exposures by this factor
        exposures = N.where(N.isfinite(scaledexp), exposures*scaledexp, 0.)

    with open(outprofile, 'w') as fout:
        incentre = True
        # print('# sbprofile_multiband.py arguments:', file=fout)
        # for a in sorted(args):
        #     print('#  %s = %s' % (a, getattr(args, a)), file=fout)

        print( '# rcentre(amin) rhalfwidth(amin) counts area(amin2) exposure sb',
               file=fout )
        for i in range(int(rmax/binf)):
            if areas[i] == 0 and incentre:
                continue
            incentre = False

            print( (0.5+i)*pixsize_arcmin,
                   0.5*pixsize_arcmin,
                   ctsum[i], areas[i], exposures[i],
                   ctsum[i]/areas[i]/exposures[i], file=fout )
