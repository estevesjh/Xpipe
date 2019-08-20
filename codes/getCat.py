# !/usr/bin/env python
# tasks: Download Chandra Observations, Reprocess data to lvl. 2, Clean flares
# Obs.: first activate ciao enviroment

from astropy.io.fits import getdata
import astropy.io.ascii as at
from astropy.table import Table
import numpy as np
import logging

## -------------------------------
## auxiliary functions
    
## -------------------------------
## Load Catalogs
def read_inputCat(catInFile, idx=None, colNames=None):
    logging.debug('Starting getCat.read_inputCat()')

    try: ## get a fits file
        h = getdata(catInFile)
    except: ## get a csv table
        h = at.read(catInFile,delimiter=',')

    d = Table(h) ## all data
    
    ## Take a sub-sample of index(idx)
    if idx is not None:
        d = d[idx]
    
    id = d[colNames[0]]
    obsids = d[colNames[1]]
    ra = d[colNames[2]]
    dec = d[colNames[3]]
    z = d[colNames[4]]

    Ncat = len(obsids)

    ## remove white spaces
    for i in range(Ncat):
        obsids[i] = obsids[i].replace(" ","")
    
    try:
        for i in range(Ncat):
            id[i] = id[i].replace(" ","")
    except:
        pass

    inputDataDict = {'ID':id,'obsids':obsids,'RA':ra,'DEC':dec,'redshift':z}
    
    logging.debug('Returning from getCat.read_inputCat()')

    return inputDataDict
    
if __name__ == '__main__':
    print('getCat.py')
    print('author: Johnny H. Esteves')

    ## Get some columns
    # if colNames is not None:
    #     inputDataDict = d[colNames]
    # else:
    #     inputDataDict = d
