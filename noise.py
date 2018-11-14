#!/usr/bin/env python
# author: Johnny Esteves, 28th September, 2018

import numpy as np
import astropy.io.fits as fits
from pycrates import *
# import matplotlib.pyplot as plt
from numpy.random import poisson

""" Module to add poisson noise to the image data

Input: Fits image
Output: bla.fits - An image with possion randon noise

"""
def noise(infits,outfile,mask):
	ctimg = read_file(infits)
	img = ctimg.get_image()
	pix = img.values
	noiseimg = poisson(pix)

	if mask != None:
		bla = read_file(mask)
		msk_values = bla.get_image().values
		msk = msk_values == 0
		noiseimg[msk] = msk_values[msk]
	
	img.values = noiseimg
	write_file(ctimg,outfile,clobber=True)

def oversampleSimple(inimage, oversample, average=False):
    """Oversample by repeating elements by oversample times."""
    os1 = N.repeat(inimage, oversample, axis=1)
    os0 = N.repeat(os1, oversample, axis=0)

    if average:
        os0 *= (1./oversample**2)

    return os0
# Scaling the image (array>1 in poisson function)
# pix = pix/np.min(pix[pix>0])
# Add a poisson noise in the image data

"""
This a module based on astropy, Obs.: don't work with ciao image!
def noise(infits):
	with pyfits.open(infits, mode='update') as hdul:

		# Assign image data to a numpy array
		img =  hdul[0].data
		# Scaling the image (array>1 in poisson function)
		img = img/np.min(img[img>0])

		# Add a poisson noise in the image data
		noise = poisson(img)
		hdul[0].data = noise

		# Saving the new image
		# hdu.writeto(outfile, overwrite=True)

		hdul.flush()

"""
