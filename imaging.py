# !/usr/bin/env python
# tasks: Flux Image, Blank Field Observations and Point Sources
# Obs.: don't forget to activate the ciao enviroment!

import astropy.io.ascii as at
from ciao_contrib.runtool import *

import sys, os
import os.path as path
import logging
import time
import subprocess
import numpy as np

## -------------------------------
## auxiliary functions

def copyFile(old_name,new_name):
    command = '{a} {b}'.format(a=old_name,b=new_name)
    os.system(command)

def psfmap(imgfilename_lis,emapfilname_lis,method='expweight',outdir='./'):
    ### 
    ''' Create a merged point source map
    input: img_lis and emap_lis should be a tupple of files paths
    methods: expweight, min; exposure map weighted psf maps and minimum PSF size '''

    nobs = len(imgfilename_lis)
    emaplist = ','.join(emapfilname_lis)
    outPSF = path.join(outdir,'merged_'+method+'.psfmap')
    psf_lis = []

    # Producing psf maps for each observation in the broad band
    for i in range(nobs):
        mkpsfmap.punlearn()
        infile = imgfilename_lis[i]
        outfile = infile.split('.img')[0]+'.psfmap'
        if not os.path.exists(outfile):
            mkpsfmap(infile,outfile,energy=2.3,spectrum="",ecf=0.9,units="arcsec",clobber=True)
        psf_lis.append(outfile)
    psf_str = ','.join(psf_lis)

    if method == 'min':
        for i in range(nobs):
            name = psf_lis[i]
            out = name.split('.psfmap')[0]+'_fov.psfmap'
            if not os.path.exists(out):
                dmimgthresh(name,out,expfile=emapfilname_lis[i],cut=1,value='INDEF',verbose=0)
        subprocess.run(['dmimgfilt','*fov.psfmap','merged_min.psfmap','min','point(0,0)','verbose=0','clob+'])
    
    if method == 'expweight':
        enu = ['(img%i*img%i)'%(i,i+nobs) for i in range(1,nobs+1)]
        div = ['img%i'%(i+nobs) for i in range(1,nobs+1)]
        op = '('+'+'.join(enu)+')/('+'+'.join(div)+')'
        if not os.path.exists(outPSF):
            subprocess.run(['dmimgcalc','infile='+psf_str+','+emaplist,'infile2=none','outfile='+outPSF,'op=imgout=%s'%(op),'verbose=0','clob+'])
    else:
        pass
    subprocess.run(['dmhedit',outPSF,'file=','op=add','key=BUNIT','value="arcsec"','verbose=0'])

def ps_merge(obsid_lis,out,energy_low=0.5,energy_high=2.,method='expweight',dirname='./'):
    ### 
    ''' Find the point sources in a broad image merged
    input: *_thresh.img','*_thresh.expmap',
    method: expweight; min
    '''
    # currentPath = os.getcwd()
    # os.chdir(dirname)
    
    ## Creating the psf map
    outNamePrefix = path.join(dirname,'img','sb_')
    imgfile = [outNamePrefix+'{0}_{1}-{2}_thresh.img'.format(obsid,energy_low,energy_high) for obsid in obsid_lis]
    expfile = [imgfile[i].split('.img')[0]+'.expmap' for i in range(len(obsid_lis))]
    
    psfmap(imgfile,expfile,method=method,outdir=dirname)

    merged_img = outNamePrefix+'{}-{}_thresh.img'.format(energy_low,energy_high)
    merged_emap = merged_img.split('.img')[0]+'.expmap'

    raw_ps = path.join(dirname,"raw_ps.fits")
    ps, ext = out, path.join(dirname,"ext.reg")

    ## Running wavdetect
    wavdetect.punlearn()
    wavdetect.infile = merged_img
    wavdetect.expfile = merged_emap
    wavdetect.scales = "2 2.828 4 5.657 8 11.314 16"
    wavdetect.ellsigma = 2.5
    wavdetect.outfile = raw_ps
    wavdetect.interdir ="./"
    wavdetect.maxiter = 5
    wavdetect.sigthresh = 1e-6
    wavdetect.scellfile = merged_img.split('.img')[0]+'_cell.fits'
    wavdetect.imagefile = merged_img.split('.img')[0]+'_recon.fits'
    wavdetect.defnbkgfile = merged_img.split('.img')[0]+'_nbkg.fits'
    wavdetect.psffile = path.join(dirname,'merged_'+method+'.psfmap')

    wavdetect(clobber=True)
    
    # Filter the point sources
    dmcopy(raw_ps+"[PSFRATIO < 1.5, SRC_SIGNIFICANCE > 3]",ps,clobber="True")
    dmcopy(raw_ps+"[PSFRATIO > 1.5, SRC_SIGNIFICANCE > 3]",ext,clobber="True")

def ps_single(img,expfile,out,dirname='./'):
    ## Creating the psf map
    psffile = img.split('.img')[0]+'.psfmap'

    mkpsfmap.punlearn()
    mkpsfmap(img,psffile,energy=2.3,spectrum="",ecf=0.9,units="arcsec",clobber=True)

    raw_ps = path.join(dirname,"raw_ps.fits")
    ps, ext = out, path.join(dirname,"ext.reg")

    ## Running wavdetect
    wavdetect.punlearn()
    wavdetect.infile = img
    wavdetect.expfile = expfile
    wavdetect.scales = "2 2.828 4 5.657 8 11.314 16"
    wavdetect.ellsigma = 2.5
    wavdetect.outfile = raw_ps
    wavdetect.interdir ="./"
    wavdetect.maxiter = 5
    wavdetect.sigthresh = 1e-6
    wavdetect.scellfile = img.split('.img')[0]+'_cell.fits'
    wavdetect.imagefile = img.split('.img')[0]+'_recon.fits'
    wavdetect.defnbkgfile = img.split('.img')[0]+'_nbkg.fits'
    wavdetect.psffile = psffile

    wavdetect(clobber=True)
    # Filter the point sources
    dmcopy(raw_ps+"[PSFRATIO < 1.5, SRC_SIGNIFICANCE > 3]",ps,clobber="True")
    dmcopy(raw_ps+"[PSFRATIO > 1.5, SRC_SIGNIFICANCE > 3]",ext,clobber="True")

def blank_field_image(obsid,evt,img,outdir='./'):
    blkevt = path.join(outdir,"{}_blank.evt".format(obsid))
    bg_img = path.join(outdir,"{}_blank_particle_bgnd.img".format(obsid))

    if not path.exists(blkevt):
        blanksky.punlearn()
        blanksky.evtfile = evt
        blanksky.outfile = blkevt
        blanksky(verbose=0,clobber=True)
    
    if not path.isfile(bg_img):
        blanksky_image.bkgfile = blkevt
        blanksky_image.outroot = blkevt.split('.evt')[0]
        blanksky_image.imgfile = img
        blanksky_image()

def merge_blank_image(img_lis,emap_lis,out_img):
    img_str = ','.join(img_lis); emap_str = ','.join(emap_lis)
    nobs = len(img_lis)
    if not path.isfile(out_img):
        enu = ['(img%i*img%i)'%(i,i+nobs) for i in range(1,nobs+1)]
        div = ['img%i'%(i+nobs) for i in range(1,nobs+1)]
        op = '('+'+'.join(enu)+')/('+'+'.join(div)+')'
        inlist = img_str+','+emap_str
        dmimgcalc(infile=inlist,infile2='none',outfile=out_img,op='imgout=%s'%(op),clobber=True)

def ccd_single(obsid,root):
    """ Find the ccd chips used in the observation (single mode)
    """
    evtgti = path.join(root,"%s"%(obsid),"repro","%s_evt_gti.fits"%(obsid))
    ra0, dec0 = [float(dmkeypar(evtgti,'RA_PNT','echo+')),float(dmkeypar(evtgti,'DEC_PNT','echo+'))]
    dmcoords(evtgti, asol="non", option="cel", ra=ra0, dec=dec0, verbose=1)
    ccd = float(dmcoords.chip_id)

    if ccd<=3:
        ccd_lis = '0:3'
    else:
        bla = dmkeypar(evtgti,"DETNAM",'echo+')
        bla = bla.split('ACIS-')[1]
        if ccd != 4:
            ccd_lis = ','.join(str(int(ccd-1+i)) for i in range(3))
        else:
            ccd_lis = ','.join(str(int(ccd+i)) for i in range(3))
    # print("%i - ccd:"%(int(obsid)),ccd_lis)
    
    return evtgti+'[ccd_id=%s]'%(ccd_lis)


def ccd_multi(obsids,root):
    """ Find the ccd chips used in the observation (multiple mode)
    """
    evt2lis = []
    for obsid in obsids:
        evtgti = path.join(root,'%i'%(int(obsid)),'repro',"%i_evt_gti.fits"%(int(obsid)))
        ra0, dec0 = [float(dmkeypar(evtgti,'RA_PNT','echo+')),float(dmkeypar(evtgti,'DEC_PNT','echo+'))]
        dmcoords(evtgti, asol="non", option="cel", ra=ra0, dec=dec0, verbose=1)
        ccd = float(dmcoords.chip_id)

        if ccd<=3:
            ccd_lis = '0:3'
        else:
            bla = dmkeypar(evtgti,"DETNAM",'echo+')
            bla = bla.split('ACIS-')[1]
            if ccd != 4:
                ccd_lis = ','.join(str(int(ccd-1+i)) for i in range(3))
            else:
                ccd_lis = ','.join(str(int(ccd+i)) for i in range(3))
        # print("%i - ccd:"%(int(obsid)),ccd_lis)

        bla = evtgti+"[ccd_id=%s]"%(ccd_lis)
        evt2lis.append(bla)

    return ', '.join(evt2lis)

def getObsid(obsid):
    lis = obsid.split(',')
    res = [int(ij) for ij in lis]
    return res

def getDir(name,path):
    nameDir = os.path.join(path,name)
    if not os.path.exists(nameDir):
        os.makedirs(nameDir)
    return nameDir

def checkDirs(dirList):
    '''Check if a list o files exists
    '''
    idx = np.empty(0,dtype=int)
    for i in range(len(dirList)):
        Dir = dirList[i]
        if path.exists(Dir):
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

def checkEvtFiles(obsid_lis,pathData='./'):
    '''Check if the event files were created.
    '''
    event_file_lis = [ path.join(pathData,"%i"%(obsid),"repro","acisf%05i_repro_evt2.fits"%(obsid)) for obsid in obsid_lis]
    idx = checkDirs(event_file_lis)
    if idx.size < len(obsid_lis):
        logging.critical('repro files was not created. Please remove the files {} and run the code again.'.format(','.join(obsid_lis)))
        exit()
    else:
        pass

def checkImg(img):
    if not path.isfile(img):
        logging.critical('Image file was not found:{}.'.format(img))
        exit()
    else:
        pass
## --------------------------------------------------------
######################## Main taks ########################
## --------------------------------------------------------

## Create Flux Images and Exposure Maps
def fluxImage(obsid,blankField=True,energy='0.7:2:1.5',binsize=2,pathData='./',outdir='./',clobber=False):
    ### http://cxc.harvard.edu/ciao/threads/merge_all/
    ### http://cxc.harvard.edu/ciao/threads/expmap_acis_multi/

    # check the OBSID input
    obsid_str, obsid_lis = checkObsid(obsid)
    nObs = len(obsid_lis)
    
    imgDir = getDir('img',outdir)

    logging.debug('Starting imaging.fluxImage(%s)'%(obsid_str))

    # check the event files
    checkEvtFiles(obsid_lis,pathData=pathData)

    elo, ehi, expMap = energy.split(':')
    
    # Define output name
    outNamePrefix = path.join(imgDir,'sb_')
    sb_img = outNamePrefix+'{0}-{1}_thresh.img'.format(elo,ehi)
    sb_img_lis = [outNamePrefix+'{0}_{1}-{2}_thresh.img'.format(obsid,elo,ehi) for obsid in obsid_lis]
    exp_lis = [sb_img_lis[i].split('.img')[0]+'.expmap' for i in range(nObs)]
    
    # do fluxImage
    checkImgFile = not path.isfile(sb_img)
    if nObs>1: ## Multiple observations
        evt2_str = ccd_multi(obsid_lis,pathData)
        evt2_lis = evt2_str.split(', ')
        
        if checkImgFile or clobber: # check image file
            merge_obs(evt2_str,outNamePrefix,band=energy,binsize=binsize,clobber=True, units='time')
    
    else: ## Single observation
        evt2_str = ccd_single(obsid_lis[0],pathData)
        
        if checkImgFile or clobber: # check image file
            fluximage(evt2_str,outNamePrefix,band=energy,binsize=binsize,clobber=True,units='time')

    # output images name
    count_img = path.join(outdir,'sb_thresh.img'); emap = path.join(outdir,'sb_thresh.expmap')
    ## copy the files
    dmcopy(sb_img,count_img,clobber=True); dmcopy(sb_img.split('.img')[0]+'.expmap',emap,clobber=True)

    # end fluxImage
    
    # start blankField
    if blankField:
        logging.debug('Starting imaging.blankField(%s)'%(obsid_str))

        # Define output name
        bg_img = path.join(outdir,'sb_blank_particle_bgnd.img')
        bg_img_lis = [path.join(imgDir,"{0}_blank_particle_bgnd.img".format(obsid)) for obsid in obsid_lis]

        # check image file
        checkImg(count_img)
        
        checkBlkimage = not path.isfile(bg_img)
        if checkBlkimage or clobber: # check blank files
            
            if nObs>1: ## Multiple observations
                for i in range(nObs):
                    blank_field_image(obsid_lis[i],evt2_lis[i],sb_img_lis[i],outdir=imgDir)
                merge_blank_image(bg_img_lis,exp_lis,bg_img) ## join blank image
            
            else: ## Single observation
                blank_field_image(obsid_lis[0],evt2_str,sb_img,outdir=imgDir)
                dmcopy(bg_img_lis[0],bg_img,clobber=True)
    # end blankField

def pointSources(obsid,name,energy_low=0.5,energy_high=2.,outdir='./',clobber=False):
    currentPath = os.getcwd()
    
    imgDir = getDir('img',outdir)
    # check the OBSID input
    obsid_str, obsid_lis = checkObsid(obsid)
    nObs = len(obsid_lis)
    
    logging.debug('Starting imaging.pointSources(%s)'%(obsid_str))

    img = path.join(outdir,'sb_thresh.img')
    expmap = img.split('.img')[0]+'.expmap' 
    psreg = path.join(outdir,"ps.reg")      #output

    # check the image file
    checkImg(img)
    if (not path.isfile(psreg)) or clobber:
        if nObs>1:
            ps_merge(obsid_lis,psreg,energy_low=energy_low,energy_high=energy_high,method='expweight',dirname=imgDir)
        else:
            ps_single(img,expmap,psreg,dirname=imgDir)

    # os.chdir(currentPath)   
    # Create Snapshots Image: check the point sources
    checkImage(img,name,psreg,imgDir=imgDir,outdir=currentPath)

def checkImage(img,name,psreg,imgDir='./',outdir='./'):
    
    checkFile = getDir('check',outdir)
    snapDir = getDir('snapshots',checkFile)

    jpgimg = path.join(snapDir,'{}_ps.jpg'.format(name))
    simg = path.join(imgDir,'smooth.img')
    
    if not os.path.exists(simg):
        csmooth(img,outfile=simg,outsig=simg.split('.img')[0]+'sig.img', outscl=simg.split('.img')[0]+'scl.img',sigmin=3, sigmax=5, sclmax=35, clobber=True)
    
    subprocess.run(['dmimg2jpg','infile=%s'%(simg),'outfile=%s'%(jpgimg),"lutfile=)ximage_lut.blue4","regionfile=region(%s)"%(psreg),"regioncolor=)colors.yellow","scalefun=log","showgrid=yes","mode=h",'clob+'])

if __name__ == '__main__':
    print('imaging.py')
    print('author: Johnny H. Esteves')
