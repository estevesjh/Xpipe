# !/usr/bin/env python
# tasks: Download Chandra Observations, Reprocess data to lvl. 2, Clean flares
# Obs.: don't forget to activate the ciao enviroment!

import astropy.io.ascii as at
from ciao_contrib.runtool import *

import sys, os
import logging
import time
import subprocess
import numpy as np

## -------------------------------
## auxiliary functions
def download_chandra_obsid(obsids):
    subprocess.run(['download_chandra_obsid',obsids])

def doFlare(evt,obsid,root='./',method='sigma'):
    ''' Clean the flares of a given observation
    input: event-file, obsid and a root file
    output: root/(obsid_evt_gti.fits, obsid_lc.fits, obsid_lc.gti)
    '''
    bla = os.path.join(root,"bla.fits")
    evtgti = os.path.join(root,"%s_evt_gti.fits"%(obsid))
    lc, lcgti = os.path.join(root,"%s_lc.fits"%(obsid)), os.path.join(root,"%s_lc.gti"%(obsid))
    dmcopy(evt+"[energy=300:12000]",bla,clobber=True)

    dmextract.punlearn()
    dmextract(bla+"[bin time=::259.28]",lc,opt="ltc1",clobber=True)

    deflare.punlearn()
    deflare(lc,lcgti,method=method)
    # subprocess.run(['punlearn','deflare'])
    # subprocess.run(['deflare',lc,lcgti,'method='+method])
    
    dmcopy(evt+'[@{0}]'.format(lcgti),evtgti)
    os.remove(bla)

def getObsid(obsid):
    lis = obsid.split(',')
    res = [int(ij) for ij in lis]
    return res

def checkFiles(evtFile):
    idx = np.empty(0,dtype=int)
    for i in range(len(dirList)):
        evt = evtFile[i]
        if os.path.isfile(evt):
            idx = np.append(i,idx)
    return idx

def checkDirs(dirList):
    '''Check if a list o files exists
    '''
    idx = np.empty(0,dtype=int)
    for i in range(len(dirList)):
        Dir = dirList[i]
        if os.path.exists(Dir):
            idx = np.append(i,idx)
    return idx

def checkObsid(obsid):
    '''It checks the obsid variable type.
       It returns in two differnt types, list and string.
    '''
    if isinstance(obsid,str):
        res_lis = getObsid(obsid)
        return obsid,res_lis

    elif isinstance(obsid,list):
        res_str = ','.join(obsid)
        return res_str,obsid

    elif isinstance(obsid,int):
        res_str = str(obsid)
        res_lis = [res_str]
        return res_str,res_lis
    else:
        logging.error('Chandra obsid={} format is not valid! Please try the following formats: int, str or list'.format(obsid))
        pass

## -------------------------------
############ Main taks ############
## -------------------------------
## Download a Chandra Observation
def down(obsid,path='./',clobber=False):
    ### http://cxc.harvard.edu/ciao/threads/archivedownload/

    obsid_str, obsid_lis = checkObsid(obsid)
    
    logging.debug('Starting preProcess.down(%s)'%(obsid_str))

    obsid_path = [os.path.join('./',str(obsid)) for obsid in obsid_lis]
    nObs = len(obsid_lis)
    
    ## Check the directory
    currentPath=os.getcwd()
    if currentPath!=path:
        os.chdir(path)
    
    idx = checkDirs(obsid_path)
    if idx.size < nObs or clobber:
        download_chandra_obsid(obsid_str)
    else:
        logging.warning('Chandra Obsid:{} already exists'.format(obsid_str))
    os.chdir(currentPath)

## -------------------------------
## Reprocess the data to level 2
def repro(obsid,path='./',clobber=False):
    ### http://cxc.harvard.edu/ciao/threads/createL2/

    ## Get the current path
    currentPath=os.getcwd()
    os.chdir(path) ## swith current directory to the download directory
    
    obsid, obsid_lis = checkObsid(obsid)
    path_lis = [os.path.join(path,str(obsid),"repro") for obsid in obsid_lis]
    nObs = len(obsid_lis)

    logging.debug('Starting preProcess.repro(%s)'%(obsid))
    # setup the chandra repro
    chandra_repro.check_vf_pha = 'yes'
    chandra_repro.set_ardlib = 'no'
    chandra_repro.clobber = True
    
    event_file_lis = [os.path.join(path,"%i"%(obsid),"repro","acisf%05i_repro_evt2.fits"%(obsid)) for obsid in obsid_lis]
    # idx = checkFiles(event_file_lis)
    # if (idx.size < nObs) or clobber:
    for (evt,obs) in zip(event_file_lis,obsid_lis):
        try:
            chandra_repro(obs,'')
        except:
            if not os.path.isfile(evt):
                try:
                    obsPath=os.path.join(path,'{}'.format(obs))
                    os.removedirs(obsPath)
                    down(obs,path=path,clobber=True)
                    chandra_repro(obs,'')
                # print('repro files was not created. Please remove the file {} and run the code again.'.format(obs))
                except:
                    logging.critical('repro files was not created. Please remove the files {} and run the code again.'.format(obs))
                    exit()
            else:
                logging.warning('{}/repro files already exists'.format(obs))
    os.chdir(currentPath)

## -------------------------------
## Clean the flares in the observations
def flare(obsid,path='./',clobber=False,method='sigma'):
    ### http://cxc.harvard.edu/ciao/threads/flare/

    # check the OBSID input
    obsid_str, obsid_lis = checkObsid(obsid)
    nObs = len(obsid_lis)
    
    logging.debug('Starting preProcess.flare(%s)'%(obsid_str))
    
    repro_lis = [os.path.join(path,"%i"%(obsid),"repro") for obsid in obsid_lis]
    event_file_lis = [os.path.join(path,"%i"%(obsid),"repro","acisf%05i_repro_evt2.fits"%(obsid)) for obsid in obsid_lis]
    evt_gti_lis = [os.path.join(path,"%i"%(obsid),"repro","%s_evt_gti.fits"%(obsid)) for obsid in obsid_lis]
    
    # check the event files
    idx = checkDirs(event_file_lis)
    if idx.size < nObs:
        mask = np.in1d(np.array(obsid_lis),np.array(obsid_lis)[idx],invert=True)
        # logging.error('{}: event files does not exists'.format(','.join(event_file_lis[mask])) )

    # check the repro files
    idx = checkDirs(evt_gti_lis)
    if idx.size < nObs or clobber:
        for i in range(nObs):
            doFlare(event_file_lis[i],obsid_lis[i],root=repro_lis[i],method='sigma')
    # else:
    #     logging.critical('repro files was not created. Please remove the files {} and run the code again.'.format(obsid))
    #     exit()
    
if __name__ == '__main__':
    print('pre-process.py')
    print('author: Johnny H. Esteves    ')
